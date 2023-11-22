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
    # This is not 100% robust but should cover MPICH & Open MPI
    for key in os.environ:
        if key.startswith("PMI_") or key.startswith("OMPI_COMM_WORLD_"):
            return True
    return False


def is_running_mpi():
    if is_running_mpiexec():
        try:
            from mpi4py import MPI  # pylint: disable=C0415
        except ImportError as e:
            raise RuntimeError(
                'it seems you are running mpiexec/mpirun but mpi4py cannot be '
                'imported, maybe you forgot to install it?') from e
    else:
        MPI = None
    return MPI


class MPIStreamHandler(logging.StreamHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MPI = is_running_mpi()
        if MPI:
            self.rank = MPI.COMM_WORLD.Get_rank()
        else:
            self.rank = 0

    def emit(self, record):
        # don't log unless I am the root process
        if self.rank == 0:
            super().emit(record)


class MPIFileHandler(logging.FileHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MPI = is_running_mpi()
        if MPI:
            self.rank = MPI.COMM_WORLD.Get_rank()
        else:
            self.rank = 0

    def emit(self, record):
        # don't log unless I am the root process
        if self.rank == 0:
            super().emit(record)


def get_comm():
    MPI = is_running_mpi()
    if MPI:
        comm = MPI.COMM_WORLD
    else:
        class comm():
            @staticmethod
            def Get_rank():
                return 0

            @staticmethod
            def Bcast(loss, root):
                pass

            @staticmethod
            def Barrier():
                pass
    return comm
