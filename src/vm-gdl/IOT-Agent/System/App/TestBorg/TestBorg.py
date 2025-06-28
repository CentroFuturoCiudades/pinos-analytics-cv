#local imports
from Generic.Global.Borg import Borg

class TestBorg(Borg):
        def __init__(self, camera):
                self.ctx = Borg._Borg__shared_state['ctx']
                self.camera = camera
        
        def __str__(self):
                print(self.__dict__)
        
        def test (self):
                self.ctx['__obj']['__log'].setLog('Starting testing')
                src = self.ctx['__obj']['__config'].get('rtsp')[self.camera]
                GP = self.ctx['__obj']['__global_procedures']
                print(self.camera)
                print(src)