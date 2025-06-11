import errno
import filecmp
import itertools
import os
import re
import subprocess
import time
from typing import Any, Optional, Union
import yaml

import zarr

# Import shared logging configuration optimized for dark themes
from .logging_config import get_mmpp_logger

# Get logger for simulation module with dark theme optimization
log = get_mmpp_logger("mmpp.simulation")


def upgrade_log_level(current_level: str, new_level: str) -> str:
    """
    Pomocnicza funkcja do "promowania" poziomu logowania.
    Zwraca "najgorszy" (najwyższy priorytetem) z podanych.
    Priorytet: ERROR > WARNING > INFO
    """
    levels = ["INFO", "WARNING", "ERROR"]
    # Wybieramy poziom, który ma "wyższy" indeks w liście
    return levels[max(levels.index(current_level), levels.index(new_level))]


class SimulationManager:
    def __init__(self, main_path: str, destination_path: str, prefix: str) -> None:
        self.main_path = main_path
        self.destination_path = destination_path
        self.prefix = prefix

    @staticmethod
    def create_path_if_not_exists(file_path: str) -> None:
        """Ensure the directory for a given file path exists."""
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    @staticmethod
    def verify_or_replace_file(new_file_path: str, existing_file_path: str) -> bool:
        """Check if a file needs replacing and replace if necessary."""
        if os.path.exists(existing_file_path):
            if filecmp.cmp(new_file_path, existing_file_path, shallow=False):
                os.remove(new_file_path)
                return True
            else:
                os.remove(existing_file_path)
        return False

    @staticmethod
    def find_status_file(path: str, sim_name: str, status: str) -> Optional[str]:
        """Locate a status file based on its name and type."""
        pattern = re.compile(rf"{sim_name}\.mx3_status\.{status}.*")
        for file in os.listdir(path):
            if pattern.match(file):
                return os.path.join(path, file)
        return None

    @staticmethod
    def get_file_status(file_path: str) -> str:
        """Determine the status of a file based on its name."""
        if ".mx3_status.lock" in file_path:
            return "locked"
        elif ".mx3_status.done" in file_path:
            return "finished"
        elif ".mx3_status.interrupted" in file_path:
            return "interrupted"
        return "unknown"

    @staticmethod
    def extract_sim_key(file_path: str) -> str:
        """Extract a concise simulation key from a file path."""
        return os.path.basename(file_path).split(".mx3_status")[0]

    @staticmethod
    def check_simulation_completion(zarr_path: str) -> tuple[bool, int]:
        """
        Check if a simulation represented by a .zarr directory is complete
        and return a tuple (is_complete, file_count).

        is_complete: bool
            True jeśli spełniony warunek >= 3000 plików w grupie 'm' oraz obecny 'end_time'.
            False w przeciwnym razie, np. gdy brakuje 'end_time', 'm' albo jest < 3000 plików.

        file_count: int
            Rzeczywista liczba plików (kroków) w grupie 'm'; 0 w razie błędu lub braku danych.
        """
        file_count = 0
        try:
            zarr_store = zarr.open(zarr_path, mode="r")
            # Sprawdzamy, czy atrybut 'end_time' istnieje
            if "end_time" not in zarr_store.attrs:
                return (False, file_count)
            # Sprawdzamy, czy istnieje grupa 'm'
            if "m" not in zarr_store:
                return (False, file_count)
            # Grupa 'm'
            m_group = zarr_store["m"]
            file_count = m_group.shape[0] if hasattr(m_group, "shape") else 0
            # Warunek 3000 plików
            if file_count >= 3000:
                return (True, file_count)
            else:
                return (False, file_count)
        except Exception:
            # Jeśli cokolwiek się wykrzaczy, zwracamy (False, 0)
            return (False, file_count)

    def submit_python_code(
        self,
        code_to_execute: str,
        last_param_name: Optional[str] = None,
        cleanup: bool = False,
        sbatch: bool = True,
        check: bool = False,
        force: bool = False,
        full_name: bool = False,
        **kwargs: Union[str, float, int],
    ) -> None:
        """
        Submit a Python simulation based on provided parameters.
        Zmodyfikowana tak, aby wszystkie komunikaty na końcu łączyć w jeden raport.
        """
        report_lines: list[str] = []
        report_log_level = "INFO"  # Poziom logowania: INFO / WARNING / ERROR
        restart_required = False  # Czy powtarzać / uruchamiać symulację?
        # skip_further_checks = (
        #     False  # Gdy np. mamy lock, done, itp. i nie chcemy powielać sprawdzeń
        # )
        sim_status_str = "unknown"
        zarr_status_str = "N/A"  # Informacja o stanie plików .zarr

        # -----------------------------
        # 2. Przygotowanie nazw i ścieżek
        # -----------------------------
        if len(kwargs) > 0 and last_param_name is None:
            last_param_name = list(kwargs.keys())[-1]

        # Build simulation parameters string (unused but kept for potential future use)
        # sim_params = "_".join(
        #     [
        #         f"{key}_{format(val, '.5g') if isinstance(val, (int, float)) else val}"
        #         for key, val in kwargs.items()
        #         if key not in [last_param_name, "i", "prefix", "template"]
        #     ]
        # )

        # par_sep = ","
        val_sep = "_"
        path = (
            f"{self.main_path}{kwargs['prefix']}/"
            + "/".join(
                [
                    f"{key}{val_sep}{format(val, '.5g') if isinstance(val, (int, float)) else val}"
                    for key, val in kwargs.items()
                    if key not in [last_param_name, "i", "prefix", "template"]
                ]
            )
            + "/"
        )

        # Wyjmujemy ostatni klucz/wartość jako klucz parametru
        last_key, last_val = kwargs.popitem()
        sim_name = f"{last_key}{val_sep}{format(last_val, '.5g')}"

        # Informacja początkowa
        report_lines.append(
            f"Checking simulation '{sim_name}', PATH: {path}{sim_name}.zarr"
        )

        self.create_path_if_not_exists(path)

        lock_file = self.find_status_file(path, sim_name, "lock")
        done_file = self.find_status_file(path, sim_name, "done")
        interrupted_file = self.find_status_file(path, sim_name, "interrupted")
        zarr_path = f"{path}{sim_name}.zarr"
        new_file_path = f"{path}{sim_name}.mx3.tmp"
        existing_file_path = f"{path}{sim_name}.mx3"

        # -----------------------------
        # 3. Logika sprawdzania plików statusu
        # -----------------------------

        # (A) lock_file => symulacja zablokowana
        if lock_file:
            sim_status_str = "locked"
            zarr_status_str = "not checked"
            # skip_further_checks = True
            # Dodatkowo możemy (opcjonalnie) sprawdzić, czy .zarr jest kompletne
            # ale z Twoich założeń: "nie ma sensu sprawdzać dalej" - więc pomijamy.

        # (B) done_file => symulacja zakończona
        elif done_file:
            sim_status_str = "finished"
            # Sprawdź jednak .zarr, bo może być niekompletne
            zarr_file_complete, zarr_file_count = self.check_simulation_completion(
                zarr_path
            )
            if zarr_file_complete:
                zarr_status_str = f"complete ({zarr_file_count} files)"
            else:
                zarr_status_str = (
                    f"incomplete ({zarr_file_count} files) => restart requaired"
                )
                restart_required = True
                report_log_level = upgrade_log_level(report_log_level, "ERROR")
            # skip_further_checks = True

        # (C) interrupted_file => symulacja przerwana
        elif interrupted_file:
            sim_status_str = "interrupted => will restart"
            zarr_status_str = "not checked"
            restart_required = True
            # skip_further_checks = True
            # Usuwamy plik, żeby móc wystartować ponownie
            os.remove(interrupted_file)

        # (D) Brak statusu => sprawdzamy .zarr (o ile istnieje)
        else:
            if os.path.exists(zarr_path):
                zarr_file_complete, zarr_file_count = self.check_simulation_completion(
                    zarr_path
                )
                if zarr_file_complete:
                    sim_status_str = "done => no status file"
                    zarr_status_str = f"complete ({zarr_file_count} files)"
                    # W takim wypadku niby nic nie trzeba robić
                else:
                    sim_status_str = "zarr incomplete => restart"
                    zarr_status_str = "incomplete => restart"
                    restart_required = True
            else:
                sim_status_str = "no status => new => restart"
                zarr_status_str = "no .zarr => start"
                restart_required = True

        # -----------------------------
        # 4. Jeśli restart_required => generujemy plik .mx3 i ewentualnie odpalamy sbatch
        # -----------------------------
        if restart_required:
            # Tworzymy nowy plik .mx3 (nadpisujemy istniejący, jeśli jest)
            with open(new_file_path, "w") as f:
                f.write(code_to_execute)

            os.rename(new_file_path, existing_file_path)

            # Uruchomienie (o ile sbatch == True)
            if sbatch:
                sim_sbatch_path = (
                    f"{self.main_path}{kwargs['prefix']}/sbatch/{sim_name}.sb"
                )
                self.create_path_if_not_exists(sim_sbatch_path)
                with open(sim_sbatch_path, "w") as f:
                    f.write(self.gen_sbatch_script(sim_name, path + sim_name))
                subprocess.call(f"sbatch < {sim_sbatch_path}", shell=True)

                sim_status_str = f"{sim_status_str} => submitted"
            else:
                report_log_level = upgrade_log_level(report_log_level, "ERROR")
                sim_status_str = f"{sim_status_str}, but not submitted (sbatch=False)"

        # -----------------------------
        # 5. Budujemy ostateczny raport (jednorazowy komunikat)
        # -----------------------------
        report_lines.append(f"Simulation status: {sim_status_str}")
        report_lines.append(f"Simulation results condition: {zarr_status_str}")

        report_message = "\n".join(report_lines)

        if report_log_level == "ERROR":
            log.error(report_message)
        elif report_log_level == "WARNING":
            log.warning(report_message)
        else:
            log.info(report_message)
        # Koniec funkcji

    def gen_sbatch_script(self, name: str, path: str) -> str:
        """Generate an sbatch script for submitting jobs."""
        mx3_file = f"{path}.mx3"
        lock_file = f"{path}.mx3_status.lock"
        done_file = f"{path}.mx3_status.done"
        interrupted_file = f"{path}.mx3_status.interrupted"
        # final_path = self.destination_path + path.replace(self.main_path, "") + ".zarr"

        return f"""#!/bin/bash -l
#SBATCH --job-name="{name}"
#SBATCH --mail-type=NONE
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --mem=149GB
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=proxima
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu39

sleep 10

# Log node name
echo "Running on node: $SLURMD_NODENAME"

nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

source /mnt/storage_3/home/kkingdyoun/.bashrc
export TMPDIR="/mnt/storage_3/home/MateuszZelent/pl0095-01/scratch/tmp/"

mv "{mx3_file}" "{lock_file}"
/mnt/storage_3/home/MateuszZelent/pl0095-01/scratch/bin/amumax -f --hide-progress-bar -o "{path}.zarr" "{lock_file}"
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "FINISHED"
    mv "{lock_file}" "{done_file}"
else
    echo "INTERRUPTED"
    mv "{lock_file}" "{interrupted_file}"

    # Check if it was a CUDA error
    if grep -q "CUDA_ERROR" "{path}.zarr/amumax.out" || nvidia-smi | grep -q "No devices were found"; then
        # Add this node to the bad nodes list if not already present
    fi
fi
    """

    @staticmethod
    def replace_variables_in_template(
        file_path: str, variables: dict[str, Union[str, float, int]]
    ) -> str:
        """Replace placeholders in a template file with actual values."""
        with open(file_path) as file:
            content = file.read()
        for key, value in variables.items():
            content = content.replace(f"{{{key}}}", str(value))
        return content

    @staticmethod
    def raw_code(*args: Any, **kwargs: Union[str, float, int]) -> str:
        """Generate the raw code by filling in a template with parameters."""
        import os

        t = kwargs["template"]
        template_path = f"{os.getcwd()}/{t}"
        return SimulationManager.replace_variables_in_template(template_path, kwargs)

    def submit_all_simulations(
        self,
        params: dict[str, Any],  # Changed from np.ndarray to Any to accept numpy arrays
        last_param_name: str,
        minsim: int = 0,
        maxsim: Optional[int] = None,
        sbatch: bool = True,
        cleanup: bool = False,
        template: str = "template.mx3",
        check: bool = False,
        force: bool = False,
        pairs: bool = False,
    ) -> None:
        """Submit all simulations based on parameter combinations.

        If pairs=False (default), generates all combinations using itertools.product.
        If pairs=True, generates paired values (par1[i], par2[i], ...) where each parameter
        must have the same number of values.
        """
        param_names = list(params.keys())

        if pairs:
            # Check that all parameter arrays have the same length
            param_lengths = [len(params[name]) for name in param_names]
            if len(set(param_lengths)) != 1:
                raise ValueError(
                    "When using pairs=True, all parameter arrays must have the same length. "
                    f"Found lengths: {dict(zip(param_names, param_lengths))}"
                )

            # Zip parameter values instead of computing product
            value_sets = zip(*[params[name] for name in param_names])
        else:
            # Original behavior - compute Cartesian product
            value_sets = itertools.product(*params.values())

        for i, values in enumerate(value_sets):
            if i < minsim:
                continue
            if maxsim is not None and i >= maxsim:
                break

            kwargs = {"prefix": self.prefix, "i": i, "template": template}
            for name, value in zip(param_names, values):
                kwargs[name] = value

            time.sleep(1)

            self.submit_python_code(
                self.raw_code(**kwargs),
                last_param_name=last_param_name,
                sbatch=sbatch,
                cleanup=cleanup,
                check=check,
                force=force,
                **kwargs,
            )


class SimulationSwapper:
    """
    Klasa do obsługi swapowania symulacji na podstawie plików konfiguracyjnych.
    Parsuje pliki YAML z parametrami i zarządza wykonaniem symulacji.
    """

    def __init__(self, config_file: str):
        """
        Inicjalizacja SwappeParsera z plikiem konfiguracyjnym.

        Args:
            config_file: Ścieżka do pliku konfiguracyjnego (YAML)
        """
        self.config_file = config_file
        self.config_data = self._load_config()
        self.parameters = self._extract_parameters()
        self.config_options = self._extract_config_options()

    def _load_config(self) -> dict[str, Any]:
        """Wczytaj plik konfiguracyjny YAML."""
        try:
            with open(self.config_file) as f:
                content = f.read()

            # Obsługa numpy w YAML - zastąp numpy calls przed parsowaniem
            content = self._preprocess_numpy_content(content)

            # Parsuj YAML
            data = yaml.safe_load(content)
            return data
        except FileNotFoundError:
            log.error(f"Configuration file not found: {self.config_file}")
            raise
        except yaml.YAMLError as e:
            log.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            log.error(f"Unexpected error loading configuration: {e}")
            raise

    def _preprocess_numpy_content(self, content: str) -> str:
        """
        Preprocess content to handle numpy expressions.
        Konwertuje wyrażenia numpy na rzeczywiste wartości.
        """
        import numpy as np

        # Znajdź wszystkie linie z numpy
        lines = content.split("\n")
        processed_lines = []

        for line in lines:
            # Jeśli linia zawiera numpy.linspace lub np.linspace
            if "np.linspace" in line or "numpy.linspace" in line:
                try:
                    # Wyciągnij część z numpy
                    if ":" in line:
                        key_part, value_part = line.split(":", 1)
                        # Wykonaj numpy expression
                        if "np.linspace" in value_part:
                            # Bezpieczne wykonanie numpy expression
                            numpy_expr = value_part.strip()
                            # Usuń komentarze
                            if "#" in numpy_expr:
                                numpy_expr = numpy_expr.split("#")[0].strip()

                            # Wykonaj wyrażenie
                            result = eval(numpy_expr, {"np": np, "numpy": np})
                            if isinstance(result, np.ndarray):
                                result = result.tolist()

                            processed_lines.append(f"{key_part}: {result}")
                        else:
                            processed_lines.append(line)
                    else:
                        processed_lines.append(line)
                except Exception as e:
                    log.warning(
                        f"Error processing numpy expression in line: {line}, error: {e}"
                    )
                    processed_lines.append(line)
            else:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _extract_parameters(self) -> dict[str, list[Any]]:
        """Wyciągnij parametry symulacji z konfiguracji."""
        parameters = {}

        for key, value in self.config_data.items():
            if key != "config" and isinstance(value, list):
                parameters[key] = value

        return parameters

    def _extract_config_options(self) -> dict[str, Any]:
        """Wyciągnij opcje konfiguracyjne."""
        default_config = {
            "last_param_name": None,
            "prefix": "v1",
            "sbatch": True,
            "full_name": False,
            "template": "template.mx3",
            "main_path": "./simulations/",
            "destination_path": "./results/",
            "minsim": 0,
            "maxsim": None,
            "pairs": False,
            "cleanup": False,
            "check": False,
            "force": False,
        }

        # Merge with user config
        if "config" in self.config_data:
            user_config = self.config_data["config"]
            default_config.update(user_config)

        return default_config

    def get_simulation_manager(self) -> "SimulationManager":
        """Stwórz instancję SimulationManager na podstawie konfiguracji."""
        return SimulationManager(
            main_path=self.config_options["main_path"],
            destination_path=self.config_options["destination_path"],
            prefix=self.config_options["prefix"],
        )

    def run_simulations(self) -> None:
        """Uruchom wszystkie symulacje na podstawie konfiguracji."""
        if not self.parameters:
            log.warning("No parameters found in configuration file")
            return

        # Sprawdź czy last_param_name jest ustawione
        last_param_name = self.config_options.get("last_param_name")
        if not last_param_name:
            log.error("last_param_name not specified in configuration")
            raise ValueError("last_param_name must be specified in config section")

        if last_param_name not in self.parameters:
            log.error(f"last_param_name '{last_param_name}' not found in parameters")
            raise ValueError(
                f"Parameter '{last_param_name}' not found in parameter list"
            )

        # Stwórz SimulationManager
        manager = self.get_simulation_manager()

        # Uruchom symulacje
        log.info(f"Starting simulations with {len(self.parameters)} parameter groups")
        log.info(f"Using last parameter: {last_param_name}")

        manager.submit_all_simulations(
            params=self.parameters,
            last_param_name=last_param_name,
            minsim=self.config_options["minsim"],
            maxsim=self.config_options["maxsim"],
            sbatch=self.config_options["sbatch"],
            cleanup=self.config_options["cleanup"],
            template=self.config_options["template"],
            check=self.config_options["check"],
            force=self.config_options["force"],
            pairs=self.config_options["pairs"],
        )

    def validate_config(self) -> list[str]:
        """
        Waliduj konfigurację i zwróć listę błędów/ostrzeżeń.

        Returns:
            Lista komunikatów o błędach/ostrzeżeniach
        """
        issues = []

        # Sprawdź czy są jakieś parametry
        if not self.parameters:
            issues.append("ERROR: No simulation parameters found")

        # Sprawdź last_param_name
        last_param_name = self.config_options.get("last_param_name")
        if not last_param_name:
            issues.append("ERROR: last_param_name not specified in config section")
        elif last_param_name not in self.parameters:
            issues.append(
                f"ERROR: last_param_name '{last_param_name}' not found in parameters"
            )

        # Sprawdź ścieżki
        main_path = self.config_options.get("main_path")
        if main_path and not os.path.exists(os.path.dirname(main_path)):
            issues.append(f"WARNING: main_path directory may not exist: {main_path}")

        # Sprawdź template
        template = self.config_options.get("template")
        if template and not os.path.exists(template):
            issues.append(f"WARNING: template file not found: {template}")

        # Sprawdź czy pairs jest sensowne
        if self.config_options.get("pairs", False):
            param_lengths = [len(values) for values in self.parameters.values()]
            if len(set(param_lengths)) > 1:
                issues.append(
                    "ERROR: When using pairs=True, all parameter arrays must have the same length"
                )

        return issues

    def get_info(self) -> dict[str, Any]:
        """Zwróć informacje o konfiguracji."""
        # Oblicz liczbę kombinacji
        if self.config_options.get("pairs", False):
            # Dla pairs - liczba wartości pierwszego parametru
            total_combinations = (
                len(list(self.parameters.values())[0]) if self.parameters else 0
            )
        else:
            # Dla product - iloczyn kartezjański
            total_combinations = 1
            for values in self.parameters.values():
                total_combinations *= len(values)

        return {
            "config_file": self.config_file,
            "parameters": list(self.parameters.keys()),
            "parameter_counts": {
                key: len(values) for key, values in self.parameters.items()
            },
            "total_combinations": total_combinations,
            "config_options": self.config_options,
            "validation_issues": self.validate_config(),
        }


# -----------------------
# PRZYKŁAD UŻYCIA
# -----------------------
# if __name__ == "__main__":
#     import itertools

#     params = {
#         "b0": np.linspace(0.0001, 0.01, 7),
#         "fcut": np.linspace(2.6, 2.8, 25),
#     }
#     num_cases = len(list(itertools.product(*params.values())))
#     log.info(f"Total number of cases: {num_cases}")

#     manager = SimulationManager(
#         main_path="/mnt/storage_2/scratch/pl0095-01/jakzwo/simulations/",
#         destination_path="/mnt/storage_2/scratch/pl0095-01/jakzwo/simulations/",
#         prefix="v5",
#     )
#     manager.submit_all_simulations(params, last_param_name="fcut", minsim=0,
#                                     maxsim=None,
#                                     sbatch=0
#                                 )
