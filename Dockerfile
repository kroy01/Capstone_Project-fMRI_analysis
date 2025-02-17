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
