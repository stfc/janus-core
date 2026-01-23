if [ ! -d tests/models/extra ]
then
    mkdir tests/models/extra
fi

(cd tests/models/extra; wget https://zenodo.org/records/16980200/files/NequIP-MP-L-0.1.nequip.zip)