#!/bin/bash

# Pfad zum Ordner mit den Rohbildern
image_dir="$PWD/img"

# Docker-Image erstellen
docker build -t piecemaker .

# Anzahl der verfügbaren Prozessorkerne ermitteln
num_cores=$(nproc)

# Durch den Ordner mit den Rohbildern iterieren
for image in "$image_dir"/*; do
    # Den Namen des Ursprungsbildes ohne Erweiterung extrahieren
    base_name=$(basename "$image" | cut -d. -f1)

    # Fünf Docker-Container für jedes Bild starten, begrenzt durch die Anzahl der Prozessorkerne
    for variant in {0..4}; do
        # Den Zielordner erstellen
        target_dir="${base_name}_${variant}"
        mkdir -p "$target_dir"

        # Den Docker-Container starten
       docker run -d -v "$image_dir":"$image_dir" -v "$PWD/out/$target_dir":"$PWD/out/$target_dir" piecemaker --dir "$PWD/out/$target_dir" --number-of-pieces "100" "$image"

        # Warten, bis die maximale Anzahl paralleler Container erreicht ist
        while [ $(docker ps -q | wc -l) -ge $num_cores ]; do
            sleep 1
        done
    done
done

# Warten, bis alle Docker-Container beendet sind
while [ $(docker ps -q | wc -l) -gt 0 ]; do
    sleep 1
done

# Alle Docker-Container des Images "piecemaker" entfernen
docker ps -a | awk '$2=="piecemaker"{print $1}' | xargs docker rm