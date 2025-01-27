#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import logging


def is_running_mpiexec():
    """
    Determines if the script is running under mpiexec.

    :return: True if running under mpiexec, False otherwise
    :rtype: bool
    """
    # This is not 100% robust but should cover MPICH & Open MPI
    for key in os.environ:
        if key.startswith("PMI_") or key.startswith("OMPI_COMM_WORLD_"):
            return True
    return False


def is_running_mpi():
    """
    Determines if the MPI environment is available and import mpi4py if so.

    :return: MPI object if available, None otherwise
    :rtype: MPI or None
    """
    if is_running_mpiexec():
        try:
            from mpi4py import MPI  # pylint: disable=C0415
        except ImportError as e:
            raise RuntimeError(
                'it seems you are running mpiexec/mpirun but mpi4py cannot be '
                'imported, maybe you forgot to install it?'
            ) from e
    else:
        MPI = None
    return MPI


class MPIStreamHandler(logging.StreamHandler):
    """
    A logging handler that only emits records from the root process in an MPI environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MPI = is_running_mpi()
        self.rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

    def emit(self, record) -> None:
        """
        Emits a log record if running on the root process.

        :param record: Log record
        :type record: Logging.LOgRecord
        """
        if self.rank == 0:
            super().emit(record)


class MPIFileHandler(logging.FileHandler):
    """
    A logging handler that only emits records to a file from the root process in an MPI environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MPI = is_running_mpi()
        self.rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

    def emit(self, record) -> None:
        """
        Emits a log record if running on the root process.

        :param record: Log record
        :type record: Logging.LOgRecord
        """
        if self.rank == 0:
            super().emit(record)


def get_comm() -> any:
    """
    Retrieves the MPI communicator if running in an MPI environment, otherwise provides a mock comm class.

    return: MPI communicator or a mock class with limited methods
    """
    mpi = is_running_mpi()
    if mpi:
        Comm = mpi.COMM_WORLD
    else:
        class Comm():
            @staticmethod
            def Get_rank():
                return 0

            @staticmethod
            def Bcast(loss, root):
                pass

            @staticmethod
            def Barrier():
                pass

    return Comm
