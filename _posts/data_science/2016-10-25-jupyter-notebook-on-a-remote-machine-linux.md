---
layout: article
title: How to work with Jupyter Notebook on a remote machine (Linux)
comments: true
categories: data_science
image:
  teaser: practical\jupyter-main-logo.svg
---

I typically use my computers at home to connect to my work computer. I setup xRDP to remote desktop into my work computer(Linux) which is OK but slow at times depending on the Internet connection. Since I usually use jypyter notebook, I wanted to be able to run a jupyter notebook server on my work computer and access it from my home computer browser. I did some search on the Internet and found a method that works, thought I'd share it here. 

1 - Open an SSH tunnel that forwards the port setup for Jupyter Notebook on the remote machine to a port on the local machine so that we can access it using our local browser

```  
hamid@local_host$ ssh user@remote_host

user@remote_host$ jupyter notebook --no-browser --port=8889
```

This runs a jupyter notebook server on the remote machine on port:8889 without opening a browser since we will use the browser on our local machine to connect to this. 


2- In a new terminal window on your local machine, SSH into the remote machine again using the following options to setup port forwarding. 

```
hamid@local_host$ ssh -N -L localhost:8888:localhost:8889 user@remote_host
```
-N options tells SSH that no commands will be run and it's useful for port forwarding, and -L lists the port forwarding configuration that we setup. 


3- Access the remote jupyter server via your local browser. Open your browser and go to:

```
localhost:8888
```

To close the SSH tunnel simply do ctrl-c. 



### Windows
- If you are using windows on your home computer but have linux on your remote machine, you can use putty to ssh into your work computer. 

- Download putty
- set ssh connection:
    + Host Name: user@IP
    + port: 22
- set putty/connections/SSH/tunnels
    + source: local port:8889
    + Destination: remote server: localhost:8080

