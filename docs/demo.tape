# Vídeo de demonstração do Formatter. Grave com: vhs docs/demo.tape (na raiz do repositório).
# MP4 para redes sociais: ffmpeg -i docs/demo.gif -c:v libopenh264 -pix_fmt yuv420p docs/demo.mp4
Output docs/demo.gif


Set Shell "bash"
Set FontSize 20
Set Width 1300
Set Height 820
Set Padding 30
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set WaitTimeout 60s

Hide
Type "source .venv/bin/activate && export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 PS1='$ ' && clear"
Enter
Show

Type "python -m formatter.predict"
Sleep 400ms
Enter
Wait+Screen /Digite um ditado/
Sleep 1s

Type "ahn eu acho que a gente pode marcar a reunião pra sexta"
Sleep 500ms
Enter
Sleep 2.5s

Type "compre o bolo na verdade não compre a torta e leve para a festa"
Sleep 500ms
Enter
Sleep 2.5s

Type "a reunião vai ser na terça não na quarta às três horas"
Sleep 500ms
Enter
Sleep 2.5s

Type "eu vou eu vou mandar o relatório hoje à noite você já viu os números"
Sleep 500ms
Enter
Sleep 2.5s

Type "o preço do arroz o preço do feijão e o preço do café subiram"
Sleep 500ms
Enter
Sleep 2.5s

Type "so um I think the tests are passing can you check the logs"
Sleep 500ms
Enter
Sleep 4s
