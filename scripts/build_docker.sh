
# Build the docker image for the interpretability toy model
docker build --pull --rm -t decisiontransformerinterpretability:latest "." --build-arg DEV_MODE=1
