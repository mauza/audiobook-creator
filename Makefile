.PHONY: build run run-gpu clean

# Variables
IMAGE_NAME = audiobook-creator
CONTAINER_NAME = audiobook-creator
AUDIOBOOKS_DIR = $(shell pwd)/audiobooks
STORIES_DIR = $(shell pwd)/stories

# Create audiobooks directory if it doesn't exist
$(shell mkdir -p $(AUDIOBOOKS_DIR))

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm \
		-v $(AUDIOBOOKS_DIR):/app/output \
		-v $(STORIES_DIR):/app/stories \
		$(IMAGE_NAME) $(ARGS)

run-gpu:
	docker run --rm \
		--gpus 1 \
		-v $(AUDIOBOOKS_DIR):/app/output \
		-v $(STORIES_DIR):/app/stories \
		$(IMAGE_NAME) $(ARGS)

clean:
	docker rmi $(IMAGE_NAME) || true

# Example usage:
# make run ARGS="convert-file stories/input.txt --voice af_heart"
# make run-gpu ARGS="convert-file stories/input.txt --voice af_heart" 