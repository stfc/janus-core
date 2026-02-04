if [ ! -d tests/models/extra ]
then
    mkdir tests/models/extra
fi

pushd tests/models/extra

    curl --output NequIP-MP-L-0.1.nequip.zip https://zenodo.org/records/16980200/files/NequIP-MP-L-0.1.nequip.zip

    curl -L --output checkpoint_sevennet_omat.pth https://github.com/MDIL-SNU/SevenNet/releases/download/v0.11.0.cp/checkpoint_sevennet_omat.pth

popd
