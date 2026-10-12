"""
Tugas Pengolahan Citra Digital
Membongkar isi foto dengan Python: metadata, data gambar (piksel/RGB), dan GPS.

Install:  pip install pillow numpy matplotlib
Pakai  :  python analisis_foto.py foto.jpg
"""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ExifTags

# Ganti dengan path foto kamu jika tidak memakai argumen terminal
PATH_FOTO = sys.argv[1] if len(sys.argv) > 1 else "IMG_20261008_195916729_HDR.jpg"


def judul(teks):
    print("\n" + "=" * 60)
    print(teks)
    print("=" * 60)


def ke_float(x):
    try:
        return float(x)
    except Exception:
        return None


def baca_exif(img):
    """Return (exif_utama, exif_detail, gps) sebagai dict bernama tag."""
    exif = img.getexif()
    utama = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}

    detail = {}
    try:
        detail = {ExifTags.TAGS.get(k, k): v for k, v in exif.get_ifd(0x8769).items()}
    except Exception:
        pass

    gps = {}
    try:
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in exif.get_ifd(0x8825).items()}
    except Exception:
        pass

    return utama, detail, gps


def dms_ke_desimal(dms, ref):
    d, m, s = (ke_float(v) for v in dms)
    nilai = d + m / 60 + s / 3600
    if ref in ("S", "W"):
        nilai = -nilai
    return nilai


# ----------------------------------------------------------------------
# 1. METADATA FOTO
# ----------------------------------------------------------------------
def analisis_metadata(path, img, utama, detail):
    judul("1. METADATA FOTO")

    print(f"Nama file        : {os.path.basename(path)}")
    print(f"Format           : {img.format}")
    print(f"Resolusi         : {img.width} x {img.height} piksel")
    print(f"Ukuran file      : {os.path.getsize(path) / 1024:.2f} KB")

    tanggal = detail.get("DateTimeOriginal") or utama.get("DateTime")
    if tanggal:
        try:
            tanggal = datetime.strptime(str(tanggal), "%Y:%m:%d %H:%M:%S").strftime(
                "%d %B %Y, %H:%M:%S"
            )
        except ValueError:
            pass
    print(f"Tanggal diambil  : {tanggal or 'Tidak tersedia'}")

    print(f"Kamera (merek)   : {utama.get('Make', 'Tidak tersedia')}")
    print(f"Kamera (model)   : {utama.get('Model', 'Tidak tersedia')}")
    print(f"Software         : {utama.get('Software', 'Tidak tersedia')}")

    iso = detail.get("ISOSpeedRatings") or detail.get("PhotographicSensitivity")
    print(f"ISO              : {iso if iso else 'Tidak tersedia'}")

    f_num = ke_float(detail.get("FNumber"))
    print(f"Aperture         : {'f/' + str(round(f_num, 1)) if f_num else 'Tidak tersedia'}")

    fl = ke_float(detail.get("FocalLength"))
    print(f"Focal length     : {str(round(fl, 1)) + ' mm' if fl else 'Tidak tersedia'}")

    eks = ke_float(detail.get("ExposureTime"))
    if eks:
        eks_txt = f"1/{round(1 / eks)} detik" if eks < 1 else f"{eks} detik"
    else:
        eks_txt = "Tidak tersedia"
    print(f"Shutter speed    : {eks_txt}")

    flash = detail.get("Flash")
    print(f"Flash            : {flash if flash is not None else 'Tidak tersedia'}")

    if not utama and not detail:
        print("\n[!] Foto ini tidak memiliki metadata EXIF "
              "(mungkin hasil screenshot/kirim lewat WhatsApp/sosmed).")


# ----------------------------------------------------------------------
# 2. DATA GAMBAR
# ----------------------------------------------------------------------
def analisis_data_gambar(img):
    judul("2. DATA GAMBAR")

    rgb = img.convert("RGB")
    arr = np.array(rgb)  # shape: (tinggi, lebar, 3)
    tinggi, lebar, kanal = arr.shape

    print(f"Mode asli        : {img.mode}")
    print(f"Ukuran gambar    : {lebar} x {tinggi} (lebar x tinggi)")
    print(f"Shape array      : {arr.shape}  -> (tinggi, lebar, kanal)")
    print(f"Jumlah piksel    : {lebar * tinggi:,} ({lebar * tinggi / 1e6:.2f} MP)")
    print(f"Tipe data        : {arr.dtype} (rentang nilai 0-255)")

    print("\nNilai piksel (pojok kiri atas, 3x3):")
    print(arr[:3, :3])

    cy, cx = tinggi // 2, lebar // 2
    print(f"\nPiksel tengah ({cx}, {cy}) -> RGB = {tuple(int(v) for v in arr[cy, cx])}")

    print("\nStatistik per kanal RGB:")
    print(f"{'Kanal':<8}{'Min':>6}{'Max':>6}{'Rata2':>10}{'Std':>10}")
    for i, nama in enumerate(["Red", "Green", "Blue"]):
        c = arr[:, :, i]
        print(f"{nama:<8}{c.min():>6}{c.max():>6}{c.mean():>10.2f}{c.std():>10.2f}")

    rata = arr.reshape(-1, 3).mean(axis=0)
    print(f"\nWarna rata-rata  : RGB({rata[0]:.0f}, {rata[1]:.0f}, {rata[2]:.0f})")
    kanal_dominan = ["Red", "Green", "Blue"][int(np.argmax(rata))]
    print(f"Kanal dominan    : {kanal_dominan}")

    gray = np.array(rgb.convert("L"))
    print(f"Kecerahan (gray) : {gray.mean():.2f} / 255")

    return arr


# ----------------------------------------------------------------------
# 3. GPS
# ----------------------------------------------------------------------
def analisis_gps(gps):
    judul("3. GPS")

    if not gps:
        print("Tidak ada data GPS pada foto ini.")
        print("(Aktifkan 'Simpan lokasi/Geotag' di kamera HP dan jangan kirim lewat WhatsApp.)")
        return

    lat = lon = None
    if "GPSLatitude" in gps and "GPSLongitude" in gps:
        lat = dms_ke_desimal(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        lon = dms_ke_desimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))

        d = [ke_float(v) for v in gps["GPSLatitude"]]
        print(f"Latitude (DMS)   : {d[0]:.0f}° {d[1]:.0f}' {d[2]:.2f}\" {gps.get('GPSLatitudeRef', '')}")
        d = [ke_float(v) for v in gps["GPSLongitude"]]
        print(f"Longitude (DMS)  : {d[0]:.0f}° {d[1]:.0f}' {d[2]:.2f}\" {gps.get('GPSLongitudeRef', '')}")
        print(f"Latitude (desimal)  : {lat:.6f}")
        print(f"Longitude (desimal) : {lon:.6f}")

    alt = ke_float(gps.get("GPSAltitude"))
    if alt is not None:
        di_bawah = gps.get("GPSAltitudeRef", 0) in (1, b"\x01")
        print(f"Ketinggian       : {-alt if di_bawah else alt:.1f} m dpl")

    if "GPSDateStamp" in gps:
        jam = gps.get("GPSTimeStamp")
        jam_txt = ":".join(f"{int(ke_float(v)):02d}" for v in jam) if jam else ""
        print(f"Waktu GPS (UTC)  : {gps['GPSDateStamp']} {jam_txt}")

    kec = ke_float(gps.get("GPSSpeed"))
    if kec is not None:
        print(f"Kecepatan        : {kec:.1f}")

    arah = ke_float(gps.get("GPSImgDirection"))
    if arah is not None:
        print(f"Arah kamera      : {arah:.1f}°")

    if lat is not None:
        print(f"\nGoogle Maps      : https://www.google.com/maps?q={lat:.6f},{lon:.6f}")
        print(f"OpenStreetMap    : https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=17/{lat:.6f}/{lon:.6f}")


# ----------------------------------------------------------------------
# 4. VISUALISASI (MATPLOTLIB)
# ----------------------------------------------------------------------
def visualisasi(arr, nama_file, simpan="hasil_analisis.png"):
    judul("4. VISUALISASI")

    gray = np.array(Image.fromarray(arr).convert("L"))
    warna = ["red", "green", "blue"]
    nama = ["Red", "Green", "Blue"]

    fig, ax = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle(f"Analisis Citra Digital - {nama_file}", fontsize=15, fontweight="bold")

    # Baris 1: gambar asli, grayscale, histogram gabungan RGB
    ax[0, 0].imshow(arr)
    ax[0, 0].set_title("Gambar Asli (RGB)")

    ax[0, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax[0, 1].set_title("Grayscale")

    for i in range(3):
        ax[0, 2].hist(arr[:, :, i].ravel(), bins=256, range=(0, 255),
                      color=warna[i], alpha=0.5, label=nama[i])
    ax[0, 2].set_title("Histogram RGB")
    ax[0, 2].set_xlabel("Nilai piksel")
    ax[0, 2].set_ylabel("Frekuensi")
    ax[0, 2].legend()

    # Baris 2: tiap kanal ditampilkan dengan warnanya
    for i in range(3):
        kanal = np.zeros_like(arr)
        kanal[:, :, i] = arr[:, :, i]
        ax[1, i].imshow(kanal)
        ax[1, i].set_title(f"Kanal {nama[i]}")

    # Baris 3: histogram per kanal
    for i in range(3):
        ax[2, i].hist(arr[:, :, i].ravel(), bins=256, range=(0, 255), color=warna[i])
        ax[2, i].set_title(f"Histogram {nama[i]} (rata2 = {arr[:, :, i].mean():.1f})")
        ax[2, i].set_xlabel("Nilai piksel")
        ax[2, i].set_ylabel("Frekuensi")

    for a in ax[:2].ravel()[:2].tolist() + list(ax[1]):
        a.axis("off")

    plt.tight_layout()
    plt.savefig(simpan, dpi=150)
    print(f"Grafik disimpan ke '{simpan}'")
    plt.show()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    if not os.path.exists(PATH_FOTO):
        print(f"File '{PATH_FOTO}' tidak ditemukan. "
              f"Jalankan: python analisis_foto.py path/ke/foto.jpg")
        sys.exit(1)

    img = Image.open(PATH_FOTO)
    utama, detail, gps = baca_exif(img)

    analisis_metadata(PATH_FOTO, img, utama, detail)
    arr = analisis_data_gambar(img)
    analisis_gps(gps)
    visualisasi(arr, os.path.basename(PATH_FOTO))


if __name__ == "__main__":
    main()