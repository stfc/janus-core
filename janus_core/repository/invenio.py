"""Repository data structure."""
from __future__ import annotations

import json
from abc import ABC
from functools import cached_property
from pathlib import Path

import requests


def _check(request: requests.Request, proc: str):
    try:
        request.raise_for_status()
    except requests.HTTPError as err:
        raise requests.HTTPError(
            f"Error while {proc}, info: {request.json()['message']}"
        ) from err

    return request

class _SubCommandHandler(ABC):
    def __init__(self, parent):
        self.parent = parent

    @property
    def url(self):
        return self.parent.url

    @property
    def api_key(self):
        return self.parent.api_key


class _File(_SubCommandHandler):
    def __init__(self, parent, name):
        super.__init__(self, parent)
        self.name = name

    @property
    def dep_id(self):
        """Get deposition ID.

        Returns
        -------
        str
            Deposition ID.
        """
        return self.parent.dep_id

    @property
    def bucket_url(self):
        """Get URL for new API file bucket.

        Returns
        -------
        str
            File bucket to ``put`` files.
        """
        return self.parent.bucket_url

    def info(self, **params) -> requests.Request:
        """Get information on a file.

        Returns
        -------
        requests.Request
            File info.
        """
        return _check(
            requests.get(
                f"{self.url}/deposit/depositions/{self.dep_id}/files/{self.name}",
                params={**params, "access_token": self.api_key},
            ),
            f"getting {self.name} file info from deposition {self.dep_id}",
        )

    def update(self, file: Path, **params) -> requests.Request:
        """Replace a file on a deposition.

        Parameters
        ----------
        file
            Source file to upload.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        data = {"name": f"{file.name}"}
        header = {"Content-Type": "application/json"}
        return _check(
            requests.put(
                f"{self.url}/deposit/depositions/{self.dep_id}/files/{self.name}",
                params={**params, "access_token": self.api_key},
                data=json.dumps(data),
                headers=header,
            ),
            f"updating {self.name} in deposition {self.dep_id}",
        )

    def download(self, dest: Path = Path(), **params) -> requests.Request:
        """Download a file from a deposition.

        Parameters
        ----------
        dest
            Folder to write files to.

        Returns
        -------
        requests.Request
            Status of operation.

        Raises
        ------
        OSError
            If destination exists and is not a directory.
        """
        info = self.info().json()
        link = info["links"]["download"]
        filename = info["filename"]

        request = _check(
            requests.get(link, params={**params, "access_token": self.api_key}),
            f"downloading file {self.name} from deposition {self.dep_id}",
        )

        dest = Path(dest)
        if dest.is_file():
            raise OSError(f"{dest} is a file which exists. Must be a directory.")

        if not dest.isdir():
            dest.mkdir(parents=True, exist_ok=True)

        with (dest / filename).open("wb") as out_file:
            out_file.write(request.content)

        return request

    def delete(self, **params) -> requests.Request:
        """Delete this file from the deposition.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        return _check(
            requests.delete(
                f"{self.url}/deposit/depositions/{self.dep_id}/files/{self.file_id}",
                params={**params, "access_token": self.api_key},
            ),
            f"deleting file {self.name} from deposition {self.dep_id}",
        )


    def upload(self, file: Path, **params) -> requests.Request:
        """Upload a file to a deposition.

        Parameters
        ----------
        file
            Path to sourcefile to upload.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        file = Path(file)

        with file.open("rb") as in_file:
            return _check(
                requests.put(
                    f"{self.bucket_url}/{self.name}",
                    params={**params, "access_token": self.api_key},
                    data=in_file,
                ),
                f"Uploading file {self.name} to deposition {self.dep_id}"
            )

class _Files(_SubCommandHandler):
    """Handler for files within a deposition."""

    def __init__(self, parent):
        super.__init__(self, parent)

    @property
    def dep_id(self) -> str:
        """Get deposition ID.

        Returns
        -------
        str
            Deposition ID.
        """
        return self.parent.dep_id

    @property
    def bucket_url(self) -> str:
        """Get URL for new API file bucket.

        Returns
        -------
        str
            File bucket to ``put`` files.
        """
        return self.parent.bucket_url

    def __getitem__(self, name) -> _File:
        return _File(self, name)

    def list(self, **params) -> requests.Request:
        """Get information about all files in deposition.

        Parameters
        ----------
        **params
            Extra params for requests.

        Returns
        -------
        requests.Request
            Information about operation state.
        """
        return _check(
            requests.get(
                f"{self.url}/deposit/depositions/{self.dep_id}/files",
                params={**params, "access_token": self.api_key},
            ),
            f"listing deposition {self.dep_id} files",
        )

    def sort(self, sorted_ids: dict[str, str], **params) -> requests.Request:
        """Re-order files in deposition.

        Parameters
        ----------
        sorted_ids
            IDs of re-sorted files.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        return _check(
            requests.put(
                f"{self.url}/deposit/depositions/{self.dep_id}/files",
                params={**params, "access_token": self.api_key},
                data=json.dumps(sorted_ids),
                headers={"Content-Type": "application/json"},
            ),
            f"sorting files for deposition {self.dep_id}"
        )

    def upload(self, files: dict[str, Path], **params) -> requests.Request:
        """Upload a set of files to a deposition.

        Parameters
        ----------
        files
            Dictionary where the key is the name for the repo,
            and the value is a path to the file to upload.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        request_list = []
        for name, file in files.items():
            file = Path(file)

            with file.open("rb") as curr_file:
                request_list.append(
                    _check(
                        requests.put(
                            f"{self.bucket_url}/{name}",
                            params={**params, "access_token": self.api_key},
                            data=curr_file,
                        ),
                        f"Uploading file {self.name} to deposition {self.dep_id}"
                    )
                )

        return request_list

    def download(self, dest: Path, **params) -> requests.Request:
        """Download all files from deposition.

        Parameters
        ----------
        dest
            Folder in which to write downloaded files.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        request = self.list(**params).json()
        files = {file["id"]: file["filename"] for file in request}

        for file in files.values():
            self[file].download(dest, **params)


class _Deposition(_SubCommandHandler):
    """Deposition handler."""

    def __init__(self, parent, dep_id):
        super().__init__(self, parent)
        self.dep_id = dep_id

    @property
    def files(self) -> _Files:
        """Get files container for this deposition.

        Returns
        -------
        _Files
            File handler.
        """
        return _Files(self)

    @cached_property
    def bucket_url(self):
        """Get URL for new API file bucket.

        Returns
        -------
        str
            File bucket to ``put`` files.
        """
        return self.get().json()["links"]["bucket"]

    def get(self, **params) -> requests.Request:
        """Get information about deposition.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        request = _check(
            requests.get(
                f"{self.url}/deposit/depositions/{self.dep_id}",
                params={**params, "access_token": self.api_key},
            ),
            f"getting deposition {self.dep_id}"
        )
        self.bucket_url = request.json()["links"]["bucket"]
        return request


    def create(self, **params) -> requests.Request:
        """Create new empty deposition.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        return _check(
            requests.post(
                f"{self.url}/deposit/depositions",
                params={**params, "access_token": self.api_key},
                json={},
                headers={"Content-Type": "application/json"},
            ),
            "creating deposition"
        )

    def update(self, data: object, **params) -> requests.Request:
        """Update deposition information.

        Parameters
        ----------
        data
            Data to be json dumped.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        return _check(
            requests.put(
                f"{self.url}/deposit/depositions/{self.dep_id}",
                params={**params, "access_token": self.api_key},
                data=json.dumps(data),
                headers={"Content-Type": "application/json"},
            ),
            f"updating deposition {self.dep_id}"
        )

    def delete(self, **params) -> requests.Request:
        """Delete deposition.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        return _check(
            requests.delete(
                f"{self.url}/deposit/depositions/{self.dep_id}",
                params={**params, "access_token": self.api_key},
            ),
            f"deleting deposition {self.dep_id}"
        )

    def publish(self, **params) -> requests.Request:
        """Publish deposition.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        return _check(
            requests.post(
                f"{self.url}/deposit/depositions/{self.dep_id}/actions/publish",
                params={**params, "access_token": self.api_key},
            ),
            f"publishing deposition {self.dep_id}",
        )

    def edit(self, **params) -> requests.Request:
        """Edit deposition details.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        return _check(
            requests.post(
                f"{self.url}/deposit/depositions/{self.dep_id}/actions/edit",
                params={**params, "access_token": self.api_key},
            ),
            f"editing deposition {self.dep_id}",
        )

    def discard(self, **params) -> requests.Request:
        """Discard deposition.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        return _check(
            requests.post(
                f"{self.url}/deposit/depositions/{self.dep_id}/actions/discard",
                params={**params, "access_token": self.api_key},
            ),
            f"discarding deposition {self.dep_id}",
        )

    def new_version(self, **params) -> requests.Request:
        """Push new version of deposition.

        Returns
        -------
        requests.Request
            Status of operation.
        """
        return _check(
            requests.post(
                f"{self.url}/deposit/depositions/{self.dep_id}/actions/newversion",
                params={**params, "access_token": self.api_key},
            ),
            f"setting new version for deposition {self.dep_id}",
        )

class _Repository(_SubCommandHandler):
    def __getitem__(self, dep_id: str) -> _Deposition:
        """Get specific deposition in repository (by id).

        Parameters
        ----------
        dep_id
            Depository ID.

        Returns
        -------
        _Deposition
            Deposition for further processing.
        """
        return _Deposition(self, dep_id)

    def list(self, **params) -> requests.Request:
        """Get information about all depositions on depository.

        Parameters
        ----------
        **params
            Extra params for requests.

        Returns
        -------
        requests.Request
            Information about operation state.
        """
        return _check(
            requests.get(
                f"{self.url}/deposit/depositions",
                params={**params, "access_token": self.api_key},
            ),
            "listing depositions"
        )


class _Records(_SubCommandHandler):
    def get(self, rec_id, **params) -> requests.Request:
        """Get information about specific record on depository.

        Parameters
        ----------
        rec_id
            ID of license to look up.
        **params
            Extra params for requests.

        Returns
        -------
        requests.Request
            Information about operation state.
        """
        return _check(
            requests.get(
                f"{self.url}/records/{rec_id}",
                params={**params, "access_token": self.api_key},
            ),
            f"getting record {rec_id}",
        )

    def list(self, **params) -> requests.Request:
        """Get information about all records on depository.

        Parameters
        ----------
        **params
            Extra params for requests.

        Returns
        -------
        requests.Request
            Information about operation state.
        """
        return _check(
            requests.get(
                f"{self.url}/records", params={**params, "access_token": self.api_key}
            ),
            "listing records",
        )

class _Licenses(_SubCommandHandler):
    def get(self, lic_id, **params) -> requests.Request:
        """Get information about specific license on depository.

        Parameters
        ----------
        lic_id
            ID of license to look up.
        **params
            Extra params for requests.

        Returns
        -------
        requests.Request
            Information about operation state.
        """
        return _check(
            requests.get(
                f"{self.url}/licenses/{lic_id}",
                params={**params, "access_token": self.api_key},
            ),
            f"getting license {lic_id}",
        )

    def list(self, **params) -> requests.Request:
        """Get information about all licenses on depository.

        Parameters
        ----------
        **params
            Extra params for requests.

        Returns
        -------
        requests.Request
            Information about operation state.
        """
        return _check(
            requests.get(
                f"{self.url}/licenses/", params={**params, "access_token": self.api_key}
            ),
            "listing licenses",
        )


class InvenioRepository:
    """Handler for Invenio-like repositories.

    Handles pushing info to e.g. Zenodo

    Parameters
    ----------
    url
        Repository URL.
    api_key : str
        API key with appropriate permissions.

    Examples
    --------
    .. code-block::

        my_repo = InvenioRepository("abc123", "companyname.website")
        my_repo.depositions["my_repo"].files["file"].upload(my_file)
        my_repo.records.get()
        my_repo.liceses.list()
    """

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key

        self.depositions = _Repository(self)
        self.records = _Records(self)
        self.licenses = _Licenses(self)
