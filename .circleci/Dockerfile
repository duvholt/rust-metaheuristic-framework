# Taken from https://abronan.com/building-a-rust-project-on-circleci/
FROM debian:stretch

# Some of the dependencies I need to build a few libraries,
# personalize to your needs. You can use multi-stage builds 
# to produce a lightweight image.
RUN apt-get update && \
    apt-get install -y curl file gcc g++ git make openssh-client \
    autoconf automake cmake libtool libcurl4-openssl-dev libssl-dev \
    libelf-dev libdw-dev binutils-dev zlib1g-dev libiberty-dev wget \
    xz-utils pkg-config python

# Install libraries/tools to cache here (OpenSSL, Capn'proto, kcov, etc.)

ENV KCOV_VERSION 34
RUN wget https://github.com/SimonKagstrom/kcov/archive/v$KCOV_VERSION.tar.gz && \
    tar xzf v$KCOV_VERSION.tar.gz && \
    rm v$KCOV_VERSION.tar.gz && \
    cd kcov-$KCOV_VERSION && \
    mkdir build && cd build && \
    cmake .. && make && make install && \
    cd ../.. && rm -rf kcov-$KCOV_VERSION

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH "$PATH:/root/.cargo/bin"
ENV RUSTFLAGS "-C link-dead-code"
ENV CFG_RELEASE_CHANNEL "nightly"

RUN rustup update && \
    rustup install nightly && \
    rustup default nightly

RUN rustup component add rustfmt-preview
RUN bash -l -c 'rustfmt -V'

RUN bash -l -c 'echo $(rustc --print sysroot)/lib >> /etc/ld.so.conf'
RUN bash -l -c 'echo /usr/local/lib >> /etc/ld.so.conf'
RUN ldconfig

RUN cargo install cargo-kcov
