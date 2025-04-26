image_name := "video-rag"
container_name := "video-rag"

build: clean
  docker build -t {{image_name}} .

clean:
  docker rm -f {{container_name}}

run: build
  docker run -it --rm --name {{container_name}} -v /mnt/c/Projects/video_rag/:/app/ -v /mnt/c/Projects/video_rag/data:/app/data {{image_name}}



