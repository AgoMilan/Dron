# 🎥 Kamera Vivotek – připojení a test

### 💡 Připojení
Adresa: `rtsp://root:asd456@192.168.0.205:554/live.sdp`

### 🧩 Test připojení
```powershell
ping 192.168.0.205
ffplay rtsp://root:asd456@192.168.0.205:554/live.sdp
```

### 🧠 PTZ Ovládání
URL šablona:
`http://root:asd456@192.168.0.205/cgi-bin/camctrl/ptz.cgi?move={cmd}`

Klávesy:
- `← / a` vlevo
- `→ / d` vpravo
- `↑ / w` nahoru
- `↓ / s` dolů
- `p` stop
