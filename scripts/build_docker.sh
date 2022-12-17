
# Build the docker image for the interpretability toy model
docker build --pull --rm -f "Dockerfile" -t decisiontransformerinterpretability:latest "." --build-arg DEV_MODE=1

