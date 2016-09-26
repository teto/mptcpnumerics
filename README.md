# mptcpnumerics
perso

# TODO
- afficher le temps de HoL blocking selon la topologie
- afficher l'overhead selon les topologies avec en abscisse le MSS et plusieurs
  courbes: une avec une option qui spawn plusieurs MSS et d'autres non




# Exemple
mptcpnumerics examples/double.json compute_constraints --sfmin fast 0.4 buffer

```
$ ./run_xp.py -pgd 
$ ./buffer_xp.py -pgd 
```

format de fichiers:


rcv_buffer/snd_buffer are in KB.
fowd/bowd are in ms
loss is in % .
cwnd is the size of the subflow congestion window in MSS
(cwnd might disappear and loss is not used yet)


