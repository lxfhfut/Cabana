import os
import time
import sys
import logging
import datetime
import colorlog
import yaml
from logging.handlers import RotatingFileHandler


class Log:
    def __init__(self):
        self.log_name = None
        self.formatter = None
        self.log_colors_config = None
        self.logger = None
        self.log_path = None

    def init_log_path(self, log_path=None):
        if log_path:
            self.log_path = log_path
        else:
            cur_path = os.path.dirname(os.path.realpath(__file__))
            self.log_path = os.path.join(cur_path, 'logs')

        # Logger
        self.logger = logging.getLogger()
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)

        # Formatter
        self.log_colors_config = {
            'DEBUG': 'green',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
        self.formatter = colorlog.ColoredFormatter(
            '%(log_color)s[%(levelname)s]- %(message)s',
            # '%(log_color)s[%(asctime)s] [%(filename)s:%(lineno)d] [%(module)s:%(funcName)s] [%(levelname)s]- %(message)s',
            log_colors=self.log_colors_config)

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        self.log_name = os.path.join(self.log_path, '%s.log' % time.strftime('%Y-%m-%d-%H-%M-%S'))

        # Clean old logs
        self.handle_logs()

        # StreamHandler for outputting to console
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(stream_handler)

        # FileHandler for writing log files
        fh = RotatingFileHandler(filename=self.log_name, mode='a', maxBytes=1024 * 1024 * 5, backupCount=5,
                                 encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

    def get_file_sorted(self):
        dir_list = os.listdir(self.log_path)
        if not dir_list:
            return
        else:
            dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(self.log_path, x)))
            return dir_list

    def timestamp_to_time(self, timestamp):
        time_struct = time.localtime(timestamp)
        return str(time.strftime('%Y-%m-%d', time_struct))

    def handle_logs(self):
        dir_list = [self.log_path]
        for direct in dir_list:
            dir_path = direct
            file_list = self.get_file_sorted()
            if file_list:
                for i in file_list:
                    file_path = os.path.join(dir_path, i)
                    t_list = self.timestamp_to_time(os.path.getctime(file_path)).split('-')
                    now_list = self.timestamp_to_time(time.time()).split('-')
                    t = datetime.datetime(int(t_list[0]), int(t_list[1]),
                                          int(t_list[2]))
                    now = datetime.datetime(int(now_list[0]), int(now_list[1]), int(now_list[2]))
                    if (now - t).days > 30:  # longer than 30 days
                        self.delete_logs(file_path)
                if len(file_list) > 30:  # log file number larger than 30
                    file_list = file_list[0:-4]
                    for i in file_list:
                        file_path = os.path.join(dir_path, i)
                        print(f"Deleting {file_path}")
                        self.delete_logs(file_path)

    def log_parameters(self, param_path):
        if not os.path.exists(param_path):
            self.logger.error("{} not exists.".format(param_path))
        else:
            str_header ="********{}********".format(os.path.basename(param_path))
            self.logger.info(str_header)
            with open(param_path, 'r') as pf:
                data = yaml.safe_load(pf)
                for line in yaml.dump(data, default_flow_style=False).split("\n"):
                    self.logger.info(line)
            str_footer = '*' * ((len(str_header) - 3) // 2) + "End" + '*' * ((len(str_header) - 3) // 2)
            self.logger.info(str_footer)

    def delete_logs(self, file_path):
        try:
            os.remove(file_path)
        except PermissionError as e:
            Log().warning('Failed to delete log file：{}'.format(e))

    def return_logger(self):
        return self.logger


Log = Log()
