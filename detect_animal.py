# detect_animal.py

import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
import sys
import platform
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deteção automática de gatos e cães com YOLOv5",
        add_help=False
    )
    parser.add_argument( "--source", type=str,
        help="‘web’ para webcam, ou caminho para imagem/vídeo ou pasta"
    )
    parser.add_argument( "--view", action="store_true",
        help="Mostrar resultado em janela OpenCV"
    )
    parser.add_argument( "--output", type=str, default=None,
        help="Nome do ficheiro de saída (será guardado em Output/). Em modo pasta, é ignorado."
    )
    parser.add_argument( "-h", "--help", action="store_true",
        help="Mostrar ajuda e sair"
    )
    return parser.parse_args()

def menu_simplificado():
    print("===== Detetor de Gatos e Cães com YOLOv5 =====")
    source = ""
    while not source:
        source = input("Fonte (‘web’ para webcam, ou caminho para imagem/vídeo ou diretório): ").strip()
    if source.lower() == "web":
        source = "0"
    ver = input("Mostrar resultado em janela? [S/n]: ").strip().lower() != "n"
    return {"source": source, "view": ver, "output": None}

def try_backend(idx, backend_flag, backend_name):
    cap = cv2.VideoCapture(idx, backend_flag)
    if cap.isOpened():
        print(f"[DEBUG] Backend {backend_name} abriu a câmara no índice {idx}.")
        return cap
    cap.release()
    print(f"[WARN] Backend {backend_name} falhou a abrir a câmara no índice {idx}.")
    return None

def open_any_camera(max_index=3):
    sistema = platform.system()
    for idx in range(max_index):
        if sistema == "Windows":
            # Primeiro MSMF, depois DSHOW
            cap = try_backend(idx, cv2.CAP_MSMF, "MSMF")
            if cap:
                return cap
            cap = try_backend(idx, cv2.CAP_DSHOW, "DSHOW")
            if cap:
                return cap
        else:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"[DEBUG] Backend Default abriu a câmara no índice {idx}.")
                return cap
            cap.release()
            print(f"[WARN] Backend Default falhou a abrir no índice {idx}.")
    return None

def process_image(path_imagem: Path, model, view: bool, output_name: str):
    frame = cv2.imread(str(path_imagem))
    if frame is None:
        print(f"Erro: não foi possível ler a imagem {path_imagem}.", file=sys.stderr)
        sys.exit(1)

    try:
        results = model(frame)
    except Exception as e:
        print(f"Erro no modelo ao processar a imagem: {e}", file=sys.stderr)
        sys.exit(1)

    detections = results.pred[0]  # tensor Nx6: [x1, y1, x2, y2, conf, cls]
    for *xyxy, conf, cls in detections.cpu().numpy():
        cls = int(cls)
        if cls == 15:
            color = (255, 0, 0)
            label = f"cat {conf:.2f}"
        elif cls == 16:
            color = (0, 255, 0)
            label = f"dog {conf:.2f}"
        else:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    n_cats = int((detections[:, -1] == 15).sum().item())
    n_dogs = int((detections[:, -1] == 16).sum().item())
    if n_cats or n_dogs:
        summary = []
        if n_cats:
            summary.append(f"{n_cats} cat(s)")
        if n_dogs:
            summary.append(f"{n_dogs} dog(s)")
        print("Detected:", ", ".join(summary))
    else:
        print("No cats or dogs detected.")

    if view:
        cv2.imshow("Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    os.makedirs("Output", exist_ok=True)
    if output_name:
        out_file = Path("Output") / output_name
    else:
        stem = path_imagem.stem
        ext = path_imagem.suffix
        out_file = Path("Output") / f"{stem}output{ext}"
    cv2.imwrite(str(out_file), frame)
    print(f"Annotated image saved to: {out_file}")

def process_video(source: str, model, view: bool, output_name: str):
    """
    Processa um vídeo (ou webcam), desenha bounding boxes em cada frame,
    exibe cada frame no tamanho original e grava o resultado em Output/.
    """
    cap = None
    webcam_mode = source.isdigit()
    if webcam_mode:
        print("[DEBUG] Tentando abrir câmara nos índices 0, 1, 2…")
        cap = open_any_camera(max_index=3)
        if not cap:
            print("Erro: não foi possível abrir nenhuma câmara nos índices tentados.", file=sys.stderr)
            sys.exit(1)
        print("[INFO] Camera opened successfully.")
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Erro: não foi possível abrir vídeo {source}.", file=sys.stderr)
            sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[DEBUG] Captured resolution: {orig_w}×{orig_h}, {fps:.2f} FPS")

    os.makedirs("Output", exist_ok=True)
    if output_name:
        out_file = Path("Output") / output_name
    else:
        if webcam_mode:
            out_file = Path("Output") / "webcamoutput.mp4"
        else:
            stem = Path(source).stem
            ext = Path(source).suffix
            out_file = Path("Output") / f"{stem}output{ext}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (orig_w, orig_h))

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Erro: a câmara abriu-se mas não devolve frames válidos.", file=sys.stderr)
        cap.release()
        sys.exit(1)
    if not webcam_mode:
        cap.release()
        cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if webcam_mode:
                time.sleep(0.01)
                continue
            else:
                break

        start = time.time()
        try:
            results = model(frame)
        except Exception as e:
            print(f"Error processing frame: {e}", file=sys.stderr)
            break

        detections = results.pred[0]
        for *xyxy, conf, cls in detections.cpu().numpy():
            cls = int(cls)
            if cls == 15:
                color = (255, 0, 0)
                label = f"cat {conf:.2f}"
            elif cls == 16:
                color = (0, 255, 0)
                label = f"dog {conf:.2f}"
            else:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if view:
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        writer.write(frame)

        elapsed = time.time() - start
        print(f"[DEBUG] Frame processed in {elapsed*1000:.1f} ms ({1/elapsed:.1f} FPS)", end="\r")

    cap.release()
    writer.release()
    if view:
        cv2.destroyAllWindows()

    print(f"\nAnnotated video saved to: {out_file}")

def process_folder(folder_path: Path, model, view: bool):
    img_exts = {".jpg", ".jpeg", ".png"}
    vid_exts = {".mp4", ".avi", ".mov"}

    for entry in sorted(folder_path.iterdir()):
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext in img_exts:
            print(f"\n[INFO] Processing IMAGE: {entry.name}")
            output_name = f"{entry.stem}output{ext}"
            process_image(entry, model, view, output_name)
        elif ext in vid_exts:
            print(f"\n[INFO] Processing VIDEO: {entry.name}")
            output_name = f"{entry.stem}output{ext}"
            process_video(str(entry), model, view, output_name)
        else:
            print(f"[SKIP] Unsupported extension, skipping: {entry.name}")

def main():
    args = parse_args()

    if len(sys.argv) == 1:
        opts = menu_simplificado()
        source = opts["source"]
        view   = opts["view"]
        output = opts["output"]
    else:
        if args.help:
            parser = argparse.ArgumentParser(
                description="Deteção automática de gatos e cães com YOLOv5"
            )
            parser.add_argument(
                "--source",
                type=str,
                required=True,
                help="‘web’ para webcam, caminho para vídeo/imagem ou diretório"
            )
            parser.add_argument(
                "--view",
                action="store_true",
                help="Mostrar resultado em janela"
            )
            parser.add_argument(
                "--output",
                type=str,
                default=None,
                help="Nome do ficheiro de saída (só para 1 ficheiro). Se for diretório, ignora."
            )
            parser.print_help()
            sys.exit(0)

        if not args.source:
            print("Erro: deve indicar --source (ficheiro, ‘web’ ou pasta).", file=sys.stderr)
            sys.exit(1)

        source = args.source
        if source.lower() == "web":
            source = "0"
        view   = args.view
        output = args.output

    print("Loading YOLOv5 model…")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    if source.isdigit():
        print("Mode: Webcam")
        process_video(source, model, view, output)

    else:
        caminho = Path(source)
        if caminho.is_dir():
            print(f"Mode: Folder. Processing all files in '{caminho}'.")
            process_folder(caminho, model, view)

        elif caminho.is_file():
            ext = caminho.suffix.lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                print("Mode: Image")
                if not output:
                    output_name = f"{caminho.stem}output{ext}"
                else:
                    output_name = output
                process_image(caminho, model, view, output_name)

            elif ext in [".mp4", ".avi", ".mov"]:
                print("Mode: Video")
                if not output:
                    output_name = f"{caminho.stem}output{ext}"
                else:
                    output_name = output
                process_video(str(caminho), model, view, output_name)

            else:
                print(f"Erro: formato não suportado: {ext}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Erro: caminho inválido: {source}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
