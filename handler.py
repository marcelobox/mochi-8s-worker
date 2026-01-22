import os, torch, subprocess, json
from diffusers import MochiPipeline

def handler(job):
    prompt = job['input']['prompt']
    out_file = '/mochi/video.mp4'

    pipe = MochiPipeline.from_pretrained(
        "genmo/mochi-1-preview",
        torch_dtype=torch.float16
    ).to("cuda")

    # 8 s = 192 frames @ 24 fps
    video = pipe(
        prompt,
        num_frames=192,
        num_inference_steps=50
    ).frames[0]

    # salva temporariamente como pngs
    os.makedirs('/tmp/frames', exist_ok=True)
    for i, frm in enumerate(video):
        frm.save(f'/tmp/frames/{i:04d}.png')

    # monta mp4
    subprocess.call([
        'ffmpeg', '-y', '-framerate', '24',
        '-i', '/tmp/frames/%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        out_file
    ])

    return {"video_url": out_file}
