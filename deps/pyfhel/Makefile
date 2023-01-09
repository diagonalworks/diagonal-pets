all: Pyfhel-3.3.2-cp39-cp39-linux_x86_64.whl

ifdef IN_DOCKER

Pyfhel-3.3.2-cp39-cp39-linux_x86_64.whl:
	pip3 wheel Pyfhel==3.3.2

else

Pyfhel-3.3.2-cp39-cp39-linux_x86_64.whl:
	docker build -t build-pyfhel .
	docker run --mount type=bind,source=${PWD},target=/build -it --name=build-pyfhel --rm --entrypoint=/usr/bin/make build-pyfhel -C /build $@ IN_DOCKER=1

endif