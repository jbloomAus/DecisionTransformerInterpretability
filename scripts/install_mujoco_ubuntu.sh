mkdir -p /root/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz
tar -xf mujoco.tar.gz -C /root/.mujoco
rm mujoco.tar.gz
echo  "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" >> /root/.bashrc
# apt install gcc
# apt-get install libglew-dev patchelf
