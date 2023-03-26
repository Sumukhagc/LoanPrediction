import sys

from src.logger import logging
def error_message(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    line_no=exc_tb.tb_lineno
    message="Error occured in file [{0}] line number [{1}] error message [{2}]" .format(file_name,line_no,str(error))
    return message

class CustomException(Exception):
    def __init__(self,error,error_detail:sys):
        super().__init__(error)
        self.error_message=error_message(error,error_detail)

    def __str__(self):
        return self.error_message
    
if __name__=='__main__':
    try:
        1/0
    except Exception as e:
        logging.info("Exception divide by zero")
        raise CustomException(e,sys)           