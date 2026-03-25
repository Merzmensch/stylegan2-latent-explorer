@echo off
title StyleGAN2 Latent Explorer
echo.
echo  Starting StyleGAN2 Latent Explorer...
echo.

call C:\Users\vladi\miniconda3\Scripts\activate.bat stylegan

cd /d C:\Users\vladi\stylegan2-explorer

python server.py --pkl models\mem.pkl

pause
```

Dann einfach **Doppelklick** auf `start.bat` — öffnet automatisch das richtige Conda Environment und startet den Server.

---

**Danach im Browser:**
```
http://localhost:5000