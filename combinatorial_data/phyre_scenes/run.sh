

conda activate phyre
make clean
make generate_tasks 
cd src/python
sudo /python3 -m phyre.generate_tasks /mnt/bn/magic/yueyang/phyre//data/task_scripts/main /mnt/bn/magic/yueyang/phyre//data/generated_tasks --save-single-pickle

tar -czvf step450k.tar.gz $(find step450k -type f -regex '.*/100[0-9][0-9]-000[0-9]0-0\.mp4')
tar -czvf step450k_train.tar.gz $(find step450k_trainset -type f -regex '.*/100[0-9][0-9]-000[0-1]0-990\.mp4')

tar -czvf step800k_train.tar.gz $(find step800k -type f -regex '.*/100[0-9][0-9]-000[0-1]0-990\.mp4')
tar -czvf step800k_eval.tar.gz $(find step800k_eval -type f -regex '.*/100[0-9][0-9]-000[0-9]0-0\.mp4')


# dit_xl_train6 1000k
cd /mnt/bn/bykang/ckpts/phyworld/phyre_combo/visdata_eval_1k/combo_4in8_train6_dit_xl
# Begin with 10003, 10005, 10016, 10023, 10024, 10053, followed by 000[0-9][0-1], end with 990
tar -czvf dit_xl_train6_step1000k_train.tar.gz $(find step1000k -type f -regex '.*/\(10003\|10005\|10016\|10023\|10024\|10053\)-000[0-9][0-1]-.*\.mp4')
tar -czvf dit_xl_train6_step1000k_eval.tar.gz $(find step1000k -type f -regex '.*/1006[0-9]-000[0-9]0-0\.mp4')

# dit_xl_train30 690k/900K/1000K
cd /mnt/bn/bykang/ckpts/phyworld/phyre_combo/visdata_eval_1k/combo_4in8_train30_dit_xl
tar -czvf dit_xl_train30_step690k_eval.tar.gz $(find Step690K_eval -type f -regex '.*/1006[0-9]-000[0-9]0-0\.mp4')
tar -czvf dit_xl_train30_step900k_eval.tar.gz $(find Step900K_eval -type f -regex '.*/1006[0-9]-000[0-9]0-0\.mp4')
tar -czvf dit_xl_train30_step1000k_eval.tar.gz $(find Step1000K_eval -type f -regex '.*/1006[0-9]-000[0-9]0-0\.mp4')

# dit_xl_train60 1000k
cd /mnt/bn/bykang/ckpts/phyworld/phyre_combo/visdata_eval_1k/combo_4in8_train60_dit_xl
tar -czvf dit_xl_train60_step1000k_train.tar.gz $(find step1000k -type f -regex '.*/100[0-5][0-9]-000[0-1]0-990\.mp4')
tar -czvf dit_xl_train60_step1000k_eval.tar.gz $(find step1000k -type f -regex '.*/1006[0-9]-000[0-9]0-0\.mp4')

# dit_xl_train30 490k
cd /mnt/bn/bykang/ckpts/phyworld/phyre_combo/visdata_eval_1k/combo_4in8_train30_dit_xl_fix
tar -czvf dit_xl_train30_fix_step490k_train30.tar.gz $(find Step490K -type f -regex '.*/100[0-5][02468]-000[0-3]0-990\.mp4')
tar -czvf dit_xl_train30_fix_step490k_eval.tar.gz $(find Step490K -type f -regex '.*/1006[0-9]-000[0-9]0-0\.mp4')


# Box2d modify b2_velocityThreshold
sudo apt-get install build-essential python-dev swig python-pygame git
git clone https://github.com/pybox2d/pybox2d
# set b2_velocityThreshold in pybox2d/Box2D/Common/b2Settings.h
python3 setup.py build
python3 setup.py install