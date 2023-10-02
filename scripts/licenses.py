import pkg_resources

def get_pkg_url(pkg):
    try:
        lines = pkg.get_metadata_lines('METADATA')
    except:
        lines = pkg.get_metadata_lines('PKG-INFO')
    
    for line in lines:
        if line.startswith('Home-page:'):
            return line[11:]
    return '(Home-page not found)'

def get_pkg_license(pkg):
    try:
        lines = pkg.get_metadata_lines('METADATA')
    except:
        lines = pkg.get_metadata_lines('PKG-INFO')

    for line in lines:
        if line.startswith('License:'):
            return line[9:]
    return '(Licence not found)'

def print_packages_and_licenses():
    for pkg in sorted(pkg_resources.working_set, key=lambda x: str(x).lower()):
        print(f"{pkg.project_name}, {pkg.version}, {get_pkg_license(pkg)}, {get_pkg_url(pkg)}, Server, No")

if __name__ == "__main__":
    print("OSS Package, Version, License, Link, Designation, Modified")
    print_packages_and_licenses()
