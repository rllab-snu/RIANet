import os

for d in os.listdir():
    if os.path.isdir(d) and d.startswith('Snaps_') and not os.path.isfile(d+'.mp4'):   
        os.system('ffmpeg -r 20 -i {}/%6d.png -c:v libx264 {}.mp4'.format(d, d))
        print('-'*50)
