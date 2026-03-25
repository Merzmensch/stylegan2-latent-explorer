@echo off
title StyleGAN2 Latent Explorer
echo.
echo  Starting StyleGAN2 Latent Explorer...
echo.

call C:\Users\YOUR-USERNAME\miniconda3\Scripts\activate.bat stylegan

cd /d C:\Users\YOUR-USERNAME\stylegan2-explorer

python server.py --pkl models\mem.pkl

pause
```

Opening Conda Enviroment.

---

Go with browser to...
```
http://localhost:5000
