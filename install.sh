#!/bin/bash

# Check if an image name and type were provided as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <image_name> <type: gpu|cpu>"
    exit 1
fi

# Assign arguments to variables
IMAGE_NAME="$1-$2"
TYPE="$2"

latest_tag=$(docker images --format "{{.Tag}}" $IMAGE_NAME | sort -V | tail -n 1)

echo "Bumping $IMAGE_NAME"

if [ -z "$latest_tag" ]; then
    echo "No tags found for image $IMAGE_NAME. Setting initial version to 1.0.0."
    new_version="1.0.0"
else
    echo "Latest version found: $latest_tag"

    # Split the latest tag into major, minor, and patch parts
    IFS='.' read -r major minor patch <<< "$latest_tag"

    # Increment the patch version
    patch=$((patch + 1))

    # Create the new version string
    new_version="$major.$minor.$patch"
    echo "Bumped version to: $new_version"
fi

echo "docker build . -t $IMAGE_NAME:$new_version -f Dockerfile.$TYPE --platform linux/amd64"
docker build . -t $IMAGE_NAME:$new_version -f Dockerfile.$TYPE --platform linux/amd64

echo "docker tag $IMAGE_NAME:$new_version scalabrese/$IMAGE_NAME:$new_version"
docker tag $IMAGE_NAME:$new_version scalabrese/$IMAGE_NAME:$new_version

echo "docker push scalabrese/$IMAGE_NAME:$new_version"
docker push scalabrese/$IMAGE_NAME:$new_version