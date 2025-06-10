#!/usr/bin/bash

if [ ! -d tests ]; then
    echo -e "Please start the script from the root of the project"
fi

python -m unittest discover -s tests -p "test_*.py"