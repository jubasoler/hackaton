import requests
import json
from PIL import Image, ImageDraw
from io import BytesIO
from dotenv import load_dotenv
from tqdm import tqdm
import os
import sys
import mediapipe as mp
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def notificar(arquivo_anexo=None):
    try:

        load_dotenv()
        SMTP = os.environ.get('SMTP')
        PORTA = os.environ.get('PORTA')
        USUARIO = os.environ.get('USUARIO')
        SENHA = os.environ.get('SENHA')
        DESTINATARIO = os.environ.get('DESTINATARIO')

        # Cria a mensagem multipart para suportar texto e anexos
        mensagem = MIMEMultipart()
        mensagem['From'] = USUARIO
        mensagem['To'] = "jubasoler@gmail.com"
        mensagem['Subject'] = "FIAP VisionGuard - ALERTA"

        # Adiciona o corpo do e-mail
        mensagem.attach(MIMEText("Alerta de objeto cortante detectado.", 'plain'))

        # Adiciona o anexo, se houver
        if arquivo_anexo:
            with open(arquivo_anexo, 'rb') as anexo:
                parte_anexo = MIMEBase('application', 'octet-stream')
                parte_anexo.set_payload(anexo.read())
                encoders.encode_base64(parte_anexo)
                parte_anexo.add_header('Content-Disposition', f'attachment; filename="{arquivo_anexo.split("/")[-1]}"')
                mensagem.attach(parte_anexo)

        # Conecta ao servidor SMTP e envia o e-mail
        with smtplib.SMTP_SSL(SMTP, PORTA) as servidor:
            servidor.login(USUARIO, SENHA)
            servidor.sendmail(USUARIO, DESTINATARIO, mensagem.as_string())
        print(f"E-mail enviado com sucesso para {DESTINATARIO}")

    except Exception as e:
        print(f"Ocorreu um erro ao enviar o e-mail: {e}")


def detect_sharp_objects(video_path, output_path, log_path):

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    file = open(log_path, "w")

    load_dotenv()
    KEY = os.environ.get('KEY')
    ENDPOINT = os.environ.get('ENDPOINT')

    headers = {
        'Prediction-Key': KEY,
        'Content-Type': 'application/octet-stream'
    }

    # Loop para processar cada frame do vídeo com barra de progresso
    for i in tqdm(range(total_frames), desc="Processando vídeo"):
        # Ler um frame do vídeo
        ret, frame = cap.read()        
        
        cv2.imwrite("frame%d.jpg" % i, frame)
        
        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        image_file = open("frame%d.jpg" % i, 'rb')

        # Enviar solicitação POST ao serviço VISION já treinado na AZURE
        response = requests.post(ENDPOINT, headers=headers, data=image_file)

        # Verificar e imprimir a resposta
        if response.status_code == 200:
            objetos = response.json()
            print(json.dumps(objetos, indent=2))
        else:
            print(f"Erro {response.status_code}: {response.json()}")
            sys.exit()

        # Iterar sobre cada predição
        for obj in objetos['predictions']:

            probability =  obj['probability']
            if float(probability) >= 0.5:
                # Obter a caixa delimitadora do objeto
                x, y, w, h = obj['boundingBox']['left'], obj['boundingBox']['top'], obj['boundingBox']['width'], obj['boundingBox']['height']
                
                file.write("Frame #{} - Predição com probabilidade de {} está localizada em -> Topo: {}, Esquerda: {}, Fundo: {}, Direita: {}. TAG: {}".format(i, probability, x, y, w, h, obj['tagName'])) 
                file.write("\n") 

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    file.write("TOTAL DE FRAMES: {}".format(total_frames))
    file.write("\n") 
    file.write("FPS: {}".format(fps))
    file.write("\n")
    notificar(log_path)
    
    # Liberar a captura de vídeo e fechar todas as janelas
    file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # Caminho para o arquivo de vídeo na mesma pasta do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, 'video2.mp4')
    output_video_path = os.path.join(script_dir, 'output_video.mp4')
    log_path = os.path.join(script_dir, 'log.txt')

    # Chamar a função para detectar emoções e reconhecer faces no vídeo, salvando o vídeo processado
    detect_sharp_objects(input_video_path, output_video_path, log_path)