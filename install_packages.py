import subprocess

req_file = 'requirements.txt'
gosdt_line = None
graphviz_line = None
normal_packages = []

with open(req_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('gosdt=='):
            gosdt_line = line
        elif line.startswith('graphviz'):
            normal_packages.append(line)
            graphviz_line = True
        else:
            normal_packages.append(line)

# Install normal packages
if normal_packages:
    subprocess.check_call(['pip', 'install', *normal_packages])

# Install gosdt without dependencies
if gosdt_line:
    subprocess.check_call(['pip', 'install', gosdt_line, '--no-deps'])

if graphviz_line:
    print("\n\n------------Attention-----------")
    print("Please install graphviz package from https://graphviz.org/download/ also and add it to environment PATH\n")
    print("-----------Warning-----------")
    print("You may be required to whitelist graphviz/bin/dot.exe file from Windows Defender/ Antivirus Softwares\n\n\n")