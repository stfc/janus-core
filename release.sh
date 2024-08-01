#!/usr/bin/env bash

version=$1

sed -i "s;^version =.*;version = \"$version\";g" pyproject.toml

git add pyproject.toml 
git commit -m "bump version for release $version"
git tag -f -a v$version -m "release $version"
git push 
git push --tags 
