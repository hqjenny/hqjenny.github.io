# Install Pytorch and Caffe2 on RISC-V Fedora
## Download the latest Fedora image

Follow the installation instructions [here](https://fedoraproject.org/wiki/Architectures/RISC-V/Installing) and download the image: 
```
wget http://fedora-riscv.tranquillity.se/kojifiles/work/tasks/4354/104354/Fedora-Developer-Rawhide-20180901.n.0-sda.raw.xz 
unxz Fedora-Developer-Rawhide-20180901.n.0-sda.raw
```

## Install
1. Run Fedora with qemu and install the prerequisite packages in Fedora 
```
dnf -y install future python-numpy openmpi protobuf python3-protobuf python3-pyyaml python3-devel python3-future
```
2. Apply patch to protobuf https://github.com/protocolbuffers/protobuf/issues/3937
3. Modify pytorch/aten/src/TH/generic/simd/simd.h, comment out the section starting at line 106 '#else // x86 ... #endif' for the following: 
```
#else   // RISCV rocket
static inline uint32_t detectHostSIMDExtensions()
{
  return SIMDExtension_DEFAULT;
}
```
4.  Install pytorch and caffe2
```
git clone https://github.com/pytorch/pytorch.git && cd pytorch
git submodule update --init --recursive
FULL_CAFFE2=1 python3 setup.py install
```
5. Test if caffe2 is installed successfully 
```
cd ~ && python3 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```
