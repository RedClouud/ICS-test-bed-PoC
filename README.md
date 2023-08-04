## About TRIST

TRIST enables the development of virtual, instantly deployable Cyber Physical Systems (CPS) which can be used to develop and test datasets using Federated Learning (FL). With this system, you can detect specific threats to CPS, such as DoS and privilege escalation (similar to Stuxnet), and can pinpoint where exactly on the [MITRE attack framework](https://attack.mitre.org/) the attacker is currently at.

TRIST is funded by [UWEcyber](http://www.cems.uwe.ac.uk/~pa-legg/uwecyber/) (thanks [UWE](https://www.uwe.ac.uk/)!)

## Technologies and features

TRIST is an open-source CPS Threat Research and Intelligence Sharing Testbed (TRIST).
TRIST aims to make CPS simulation and dataset development easy by providing the following features:

- CPS development and deployment
- CPS dataset development and testing
- CPS attack simulation and identification
- Full data analytics tools

It uses multiple technolgies to achieve this including [ELK](https://www.elastic.co/), [Docker](https://www.docker.com/), [Flower](https://flower.dev/), and more.

TRIST is part inspired by [MiniCPS](https://minicps.readthedocs.io/), a CPS simulation tool built on top of [Mininet](http://mininet.org/). However, TRIST takes this one step further by implementing any CPS into portable and instantly deployable Docker containers. This means no installation, configuration, or setup is requried. Simply download the Docker image and press go. It is then as simple as accessing the TRIST Dashboard via a web browser to access all functionality.

We hope you enjoy using TRIST!

---

# Federated Learning implemented into Docker

1. Clone this repo
2. Grab the Docker image using `$ docker pull redclouud/trist`
3. Execute `$ python2 init.py`

# Run

1. Navigate into TRIST
2. Execute `$ docker-compose up`

Enjoy!

## How to

1. Execute build.sh: ensures that the required Docker images are built

   ```bash
   $ bash build.sh
   ```

2. Execute the docker-compose file: builds the FL environment and begins learning

   ```bash
   $ cd learn
   $ docker-compose up -d
   ```

You should now have three containers running:

- One container acting as the FL server
- Two containers acting as the FL clients

---

## TODO

- [ ] Upload the Docker images so that they are available without needing to build

## Contact

If you have any enquiries, such as problems or suggestions, then email us at contact dot q65xp at slmail dot me
