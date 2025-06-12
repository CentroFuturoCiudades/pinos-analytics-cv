import subprocess

def run_script(script_path):
    try:
        # Ejecuta el script y espera a que termine
        subprocess.run(["python", script_path], check=True)
        print(f"Successfully executed {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_path}: {e}")

if __name__ == "__main__":
    # Ruta para descargar los videos (solo si inferred=false)
    download_videos = "./video_downloader.py"

    # Ruta para procesar los videos con yolo y subirlos a la base de datos (todas las camaras)
    process_videos = "./video_processing.py"

    # Ruta para el hacer spatial join (duración en galerias - solo camaras 4 y 5)
    spatial_join_galeries = "./durations_spatial_join.py"

    # Ejecutar el script para descargar videos
    print("Downloading videos...")
    run_script(download_videos)

    # Ejecutar el script de procesamiento
    print("Processing video entities and inputting them in database...")
    run_script(process_videos)

    # Ejecutar el script de spatial join
    print("Perfoming spatial join...")
    run_script(spatial_join_galeries)