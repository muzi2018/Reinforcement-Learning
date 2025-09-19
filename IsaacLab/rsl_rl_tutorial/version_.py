from packaging import version

v1 = version.parse("3.9.7")
v2 = version.parse("3.10.1")

if v2 > v1:
    print(f"{v2} is newer than {v1}")
