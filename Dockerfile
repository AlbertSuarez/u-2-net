FROM ubuntu:18.04

# Add content
ADD . /srv/u-2-net
WORKDIR /srv/u-2-net

# Update and install Python3
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-virtualenv

# Set up Virutal environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
RUN pip install --upgrade pip
RUN pip3 install -r requirements.lock

# Run
CMD python3 -m src.run ${INPUT_PATH} ${OUTPUT_PATH}
