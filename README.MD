# Current Status
* only fsl and dcm2niix has been successfully installed in container.
* dockerfile content:
```dockerfile
FROM ubuntu:22.04

ENV FSLDIR          "/usr/local/fsl"
ENV DEBIAN_FRONTEND "noninteractive"
ENV LANG            "en_GB.UTF-8"

RUN apt update  -y && \
    apt upgrade -y && \
    apt install -y    \
      python3         \
      wget            \
      file            \
      dc              \
      mesa-utils      \
      pulseaudio      \
      libquadmath0    \
      libgtk2.0-0     \
      firefox         \
      libgomp1        \
      dcm2niix

RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py
RUN python3 ./fslinstaller.py -d /usr/local/fsl/

ENTRYPOINT [ "bash", "-c", "source /usr/local/fsl/etc/fslconf/fsl.sh && exec bash" ]


```

to run and execute dicom conversion following commands need to be run on shell
1. build the image
    ```shell
    podman build -t fmri-dicom-converter .
    ```
2. set xhost for using fsleyes with gui
    ```shell
    xhost +local:root
    ```
3. run it with appropriate parameters.
   (optional : set output file name as env variable.)
    ```shell
    podman run --rm -it \
    > --net=host \
    > -e DISPLAY=$DISPLAY \
    > -v /tmp/.X11-unix:/tmp.X11-unix \
    > -v $(pwd)/I1270009:/input \
    > -v $(pwd)/output:/output \
    > -e OUTPUT_NAME="converted_output" \
    > fmri-dicom-converter
    ```
4. after running the container shell will be available with root privileges. use dcm2nixx for conversion.
    ```shell
    dcm2niix -o /output -f "${OUTPUT_NAME}" /input
    ```
5. you can use fsleyes to view the output file created
    ```shell
    fsleyes output/converted_output.nii
    ```