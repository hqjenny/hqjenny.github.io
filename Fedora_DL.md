# Install Pytorch and Caffe2 on RISC-V Fedora
## Download the latest Fedora image

Follow the installation instructions [here](https://fedoraproject.org/wiki/Architectures/RISC-V/Installing) and download the image: 
```
wget http://fedora-riscv.tranquillity.se/kojifiles/work/tasks/4354/104354/Fedora-Developer-Rawhide-20180901.n.0-sda.raw.xz 
unxz Fedora-Developer-Rawhide-20180901.n.0-sda.raw
```

Qemu script
```
qemu-system-riscv64 \
    -nographic \
    -machine virt \
    -smp 4 \ 
    -m 16G \
    -kernel bbl-vmlinux\
    -object rng-random,filename=/dev/urandom,id=rng0 \
    -device virtio-rng-device,rng=rng0 \
    -append "console=ttyS0 ro root=/dev/vda1" \
    -device virtio-blk-device,drive=hd0 \
    -drive file=Fedora-Developer-Rawhide-20180901.n.0-sda.raw,format=raw,id=hd0 \
    -device virtio-net-device,netdev=usernet \
    -netdev user,id=usernet,hostfwd=tcp::10000-:22
```

## Install
1. Run Fedora with qemu and install the prerequisite packages in Fedora 
```
dnf -y install future python-numpy openmpi protobuf python3-protobuf python3-pyyaml python3-devel python3-future
```
2. Git clone pytorch repo
```
git clone https://github.com/pytorch/pytorch.git && cd pytorch
git submodule update --init --recursive
```
3. Apply patch to protobuf https://github.com/protocolbuffers/protobuf/issues/3937
4. Modify pytorch/aten/src/TH/generic/simd/simd.h, comment out the section starting at line 106 '#else // x86 ... #endif' for the following: 
```
#else   // RISCV rocket
static inline uint32_t detectHostSIMDExtensions()
{
  return SIMDExtension_DEFAULT;
}
```
5. Add the following code block to line 173
```
//#error You need to define CycleTimer for your OS and CPU
uint64_t tsc;
asm volatile("rdcycle %0 " : "=r"(tsc));
return tsc;
```

6. Install pytorch and caffe2
```
FULL_CAFFE2=1 python3 setup.py install
```
7. Test if caffe2 is installed successfully 
```
cd ~ && python3 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```
