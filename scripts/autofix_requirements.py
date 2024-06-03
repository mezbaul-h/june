import os
import pathlib
import subprocess
import tempfile
import typing


def read_requirements(requirements_file_path: pathlib.Path) -> typing.Dict[str, str]:
    """
    Reads the requirements from the specified file and returns a dictionary
    mapping package names to their versions.
    """
    with open(requirements_file_path, mode="r") as requirements_file:
        requirements = requirements_file.read().splitlines()
        requirements_map = {}

        for requirement in requirements:
            try:
                package_name, version = requirement.split("==")
                requirements_map[package_name] = version
            except ValueError:
                requirements_map[requirement] = ""

    return requirements_map


def write_temp_requirements(requirements_map: typing.Dict[str, str]) -> str:
    """
    Writes the package names (without versions) to a temporary file
    and returns the path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write("\n".join(requirements_map.keys()))

    return temp_file_path


def run_pip_install(temp_file_path: str, requirements_map: typing.Dict[str, str]):
    """
    Runs 'pip install -r temp_file_path' and updates the requirements_map
    with the installed package versions.
    """
    command = ["pip", "install", "-r", temp_file_path]

    try:
        process = subprocess.Popen(
            " ".join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Capture and print the output line-by-line
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output and output.startswith("Successfully installed"):
                for item in output.strip().split(" "):
                    if "-" in item:
                        package_name, version = item.split("-")
                        requirements_map[package_name] = version

        # Capture any remaining output after the process completes
        rc = process.poll()
        if rc is not None:
            _ = process.stdout.read()

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")


def update_requirements_file(requirements_file_path: pathlib.Path, requirements_map: typing.Dict[str, str]):
    """
    Writes the updated requirements (with versions) back to the original requirements file.
    """
    with open(requirements_file_path, mode="w") as requirements_file:
        requirements_file.writelines([f"{k}=={v}\n" for k, v in requirements_map.items()])


def main():
    current_dir = pathlib.Path(__file__).parent.absolute()
    project_dir = current_dir.parent
    requirements_file_path = project_dir / "requirements.txt"

    # Step 1: Read the requirements
    requirements_map = read_requirements(requirements_file_path)

    # Step 2: Create a temporary file with package names
    temp_file_path = write_temp_requirements(requirements_map)

    # Step 3: Run pip install and update the requirements map with installed versions
    run_pip_install(temp_file_path, requirements_map)

    # Step 4: Update the original requirements file with the new versions
    update_requirements_file(requirements_file_path, requirements_map)

    # Clean up the temporary file
    os.remove(temp_file_path)
    print(f"Temporary file {temp_file_path} deleted.")


if __name__ == "__main__":
    main()
