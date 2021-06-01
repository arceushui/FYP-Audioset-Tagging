import sys
import tkinter as tk
from tkinter import ttk as ttk
from tkinter import font
from tkinter import filedialog
from threading import Thread
from time import sleep
from types import SimpleNamespace

import audio_detection

from inference import audio_tagging

class DemonstratorGUI(ttk.Frame):
    
    def __init__(self, master=None,number_of_results=None):
        super().__init__(master)
        self.master = master
        
        self.masterFrameSwitchWidth = 560
        self.masterFrameSwitchHeight = 840
        
        if self.winfo_screenheight() < 900 or self.winfo_screenwidth() < 600 :
            startUpSize = 'small'
        else:
            startUpSize = 'normal'
                
            
        self.banner = tk.PhotoImage(file='./images/msos_banner.ppm') 
        self.bannerSmall = tk.PhotoImage(file='./images/msos_banner_small.ppm') 
        
        self.demoStyle = ttk.Style()
        self.font = None    
        
        
        self.numberOfResults = int(number_of_results)
        
        self.posthocProb = [None] * self.numberOfResults
        for k in range(self.numberOfResults):
            self.posthocProb[k] = tk.IntVar()
            self.posthocProb[k].set(0)
        
        self.resultLabel = [None] * self.numberOfResults
        for q in range(self.numberOfResults):
            self.resultLabel[q] = tk.StringVar()
            tempPHStr = '-----'
            placeholderStr='{0:<15.15}'.format(tempPHStr)
            self.resultLabel[q].set(placeholderStr)
                
            
        
        
        self.define_styles()    
        self['style'] = 'ourMaster.TFrame'      
        self.pack(fill="both",expand=True)
        
        self.create_widgets()
        
        self.overallSize = startUpSize
        self.image_selection(startUpSize)
        self.apply_styles(startUpSize)
        self.apply_packing(startUpSize)
        
        self.bind("<Configure>", self.track_overall_size)
        
        self.aud_detec = audio_detection.AudioDetection(self.numberOfResults)
        
        #self.update()
        #print('Win width = ' + str(self.winfo_width()))
        
        
    def create_widgets(self):
        
        self.procStatus = False
        
        
        # ------------------ Banner -------------------------
        self.bannerFrame = ttk.Frame(self)
        self.bannerLabel = ttk.Label(self.bannerFrame)
        
        # ------------------ Title -------------------------
        self.titleFrame = ttk.Frame(self)
        self.titleLabel = ttk.Label(self,text='Sound Recognition')

        # ------------------ Load audio file -------------------------
        self.loadFileFrame = ttk.Frame(self,style='loadFile.TFrame')
        self.loadButton = ttk.Button(self.loadFileFrame, text="Load File")
        self.loadButton['command'] = self.load_audio_file
        self.filename = ""

        # ------------------ Start & stop button -----------
        self.startStopFrame = ttk.Frame(self,style='startStop.TFrame')
        self.startButton = ttk.Button(self.startStopFrame,text='Start')
        self.startButton['command'] = self.start_stop_recording_and_analysis

        # ------------------ Results master frame -----------
        self.resultsOverallFrame = ttk.Frame(self,style='resultsOverall.TFrame')
        
        # ------------------ Results headings -----------
        self.resultsHeadingFrame = ttk.Frame(self.resultsOverallFrame)
        self.resRankLabel = ttk.Label(self.resultsHeadingFrame,text='Rank')
        self.resLabLabel = ttk.Label(self.resultsHeadingFrame,text='{0:<12}'.format('Label'))
        self.resConfLabel = ttk.Label(self.resultsHeadingFrame,text='Confidence')
        
        # ------------------ Results entries -----------
        self.resultsRowFrame = [None] * self.numberOfResults
        self.resRankEntryLabel = [None] * self.numberOfResults
        self.resLabelEntryLabel = [None] * self.numberOfResults
        self.resConfEntryLabel = [None] * self.numberOfResults              
        for n in range(self.numberOfResults):
            
            # Row frame
            self.resultsRowFrame[n] = ttk.Frame(self.resultsOverallFrame)
            
            # Rank label
            self.resRankEntryLabel[n] = ttk.Label(self.resultsRowFrame[n],text='{0}'.format(n+1))
        
            # Labels of recognised sounds
            self.resLabelEntryLabel[n] = ttk.Label(self.resultsRowFrame[n],textvariable=self.resultLabel[n])
            
            # Confidence bars
            self.resConfEntryLabel[n] = ttk.Progressbar(self.resultsRowFrame[n],orient='horizontal',
                                  mode='determinate',maximum=1,variable=self.posthocProb[n]) 
            
        
        # ------------------ Exit button -----------
        self.exitFrame = ttk.Frame(self)
        self.exitButton = ttk.Button(self.exitFrame,text='Exit')
        self.exitButton['command'] = self.exit_demo
        
        


    def define_styles(self):
        
        self.font = self.ourFonts()
        
        # Frames
        self.demoStyle.configure('ourMaster.TFrame', background='white')
        self.demoStyle.configure('banner.TFrame', background='white')
        self.demoStyle.configure('title.TFrame', background='white')
        self.demoStyle.configure('startStop.TFrame', background='white')
        self.demoStyle.configure('loadFile.TFrame', background='white')
        self.demoStyle.configure('resultsOverall.TFrame', background='white')
        self.demoStyle.configure('resultsHeading.TFrame', background='white')
        self.demoStyle.configure('resultsRow.TFrame', background='white')
        self.demoStyle.configure('exit.TFrame', background='white')
        
        # Labels
        self.demoStyle.configure('banner.TLabel', background='white',font=self.font.title)
        self.demoStyle.configure('title.TLabel', background='white',font=self.font.title)
        self.demoStyle.configure('titleSmall.TLabel', background='white',font=self.font.titleSmall)
        self.demoStyle.configure('headResults.TLabel', background='white',font=self.font.labelHead)
        self.demoStyle.configure('headResultsSmall.TLabel', background='white',font=self.font.labelHeadSmall)
        self.demoStyle.configure('entryResults.TLabel',background='white',font=self.font.labelEntry)
        self.demoStyle.configure('entryResultsSmall.TLabel',background='white',font=self.font.labelEntrySmall)
        self.demoStyle.configure('entryLabResults.TLabel',background='white',font=self.font.labelEntryLab)
        self.demoStyle.configure('entryLabResultsSmall.TLabel',background='white',font=self.font.labelEntryLabSmall)
        
        # Buttons
        self.demoStyle.configure('startStop.TButton',foreground='black',background='white',font=self.font.button)
        self.demoStyle.map("startStop.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('startStopSmall.TButton',foreground='black',background='white',font=self.font.buttonSmall)
        self.demoStyle.map("startStopSmall.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('load.TButton',foreground='black',background='white',font=self.font.button)
        self.demoStyle.map("load.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('loadSmall.TButton',foreground='black',background='white',font=self.font.buttonSmall)
        self.demoStyle.map("loadSmall.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('exit.TButton', foreground='#7d150e', background='white', font=self.font.button)
        self.demoStyle.map('exit.TButton',background=[('active', '#7cb669')])
        self.demoStyle.configure('exitSmall.TButton', foreground='#7d150e', background='white', font=self.font.buttonSmall)
        self.demoStyle.map('exitSmall.TButton',background=[('active', '#7cb669')])

        # Progress bar
        self.demoStyle.configure("entryConf.Horizontal.TProgressbar", foreground='black', background='#7d89af',
                            troughcolor='white',relief='sunken')
        self.demoStyle.configure("entryConfSmall.Horizontal.TProgressbar", foreground='black', background='#7d89af',
                            troughcolor='white',relief='sunken')
        

    def apply_styles(self,overall_size):
        
        if overall_size == 'normal':        
            self.bannerFrame['style'] = 'banner.TFrame'
            self.bannerLabel['style'] = 'banner.TLabel'
            self.titleFrame['style'] = 'title.TFrame'
            self.titleLabel['style'] = 'title.TLabel'
            self.startButton['style'] = 'startStop.TButton'
            self.loadButton['style'] = 'loadFile.TButton'
            self.resultsOverallFrame['style'] = 'resultsOverall.TFrame'
            self.resultsHeadingFrame['style'] = 'resultsHeading.TFrame'
            self.resRankLabel['style'] = 'headResults.TLabel'
            self.resLabLabel['style'] = 'headResults.TLabel'
            self.resConfLabel['style'] = 'headResults.TLabel'
            for n in range(self.numberOfResults):
                self.resultsRowFrame[n]['style'] = 'resultsRow.TFrame'
                self.resRankEntryLabel[n]['style'] = 'entryResults.TLabel'
                self.resLabelEntryLabel[n]['style'] = 'entryLabResults.TLabel'
                self.resConfEntryLabel[n]['style'] = 'entryConf.Horizontal.TProgressbar'
                self.resConfEntryLabel[n]['length'] = 200
            self.exitFrame['style'] = 'exit.TFrame'    
            self.exitButton['style'] = 'exit.TButton'
                           
        elif overall_size == 'small':   
            self.bannerFrame['style'] = 'banner.TFrame'
            self.bannerLabel['style'] = 'banner.TLabel'
            self.titleFrame['style'] = 'title.TFrame'
            self.titleLabel['style'] = 'titleSmall.TLabel'
            self.startButton['style'] = 'startStopSmall.TButton'
            self.loadButton['style'] = 'loadFileSmall.TButton'
            self.resultsOverallFrame['style'] = 'resultsOverall.TFrame'
            self.resultsHeadingFrame['style'] = 'resultsHeading.TFrame'
            self.resRankLabel['style'] = 'headResultsSmall.TLabel'
            self.resLabLabel['style'] = 'headResultsSmall.TLabel'
            self.resConfLabel['style'] = 'headResultsSmall.TLabel'
            for n in range(self.numberOfResults):
                self.resultsRowFrame[n]['style'] = 'resultsRow.TFrame'
                self.resRankEntryLabel[n]['style'] = 'entryResultsSmall.TLabel'
                self.resLabelEntryLabel[n]['style'] = 'entryLabResultsSmall.TLabel'
                self.resConfEntryLabel[n]['style'] = 'entryConfSmall.Horizontal.TProgressbar'
                self.resConfEntryLabel[n]['length'] = 100
            self.exitFrame['style'] = 'exit.TFrame'    
            self.exitButton['style'] = 'exitSmall.TButton'
                      
        else:
            raise Exception('Unkown overall size {0}!'.format(overall_size))    
            
            
            
    def apply_packing(self,overall_size):
        
        if overall_size == 'normal':
            self.bannerFrame.pack(side='top', fill="both", expand=True, padx=0, pady=0) 
            self.bannerLabel.pack(side='top', expand=True, padx=0, pady=0)
            self.titleFrame.pack(side='top', fill="both", expand=True, padx=10, pady=10) 
            self.titleLabel.pack(side='top', expand=True, padx=20, pady=20)
            self.startStopFrame.pack(side='top', expand=True, padx=10, pady=10) 
            self.loadFileFrame.pack(side='top', expand=True, padx=10, pady=10) 
            self.startButton.pack(side='left', expand=True, padx=30, pady=10, ipadx=80, ipady=20)
            self.loadButton.pack(side='left', expand=True, padx=30, pady=10, ipadx=80, ipady=20)
            self.resultsOverallFrame.pack(expand=True, padx=20, pady=0)
            self.resultsHeadingFrame.pack(side='top', expand=True, padx=0, pady=10)
            self.resRankLabel.pack(side='left',expand=True, padx=0, pady=20)
            self.resLabLabel.pack(side='left',expand=True, padx=40, pady=20)
            self.resConfLabel.pack(side='left',expand=True, padx=30, pady=20)
            for n in range(self.numberOfResults):
                self.resultsRowFrame[n].pack(side='top', expand=True, padx=20, pady=10)
                self.resRankEntryLabel[n].pack(side='left', expand=True, padx=10, pady=5)
                self.resLabelEntryLabel[n].pack(side='left', expand=True, padx=10, pady=5)
                self.resConfEntryLabel[n].pack(side='left', expand=True, padx=20, pady=10)
            self.exitButton.pack(expand=True, padx=20, pady=40, ipadx=80, ipady=20)
            self.exitFrame.pack(side='top', fill="both", expand=True, padx=0, pady=0)
            
        elif overall_size == 'small':
            self.bannerFrame.pack(side='top', fill="both", expand=True, padx=0, pady=0) 
            self.bannerLabel.pack(side='top', expand=True, padx=0, pady=0)
            self.titleFrame.pack(side='top', fill="both", expand=True, padx=5, pady=5) 
            self.titleLabel.pack(side='top', expand=True, padx=10, pady=10)
            self.startStopFrame.pack(side='top', expand=True, padx=5, pady=5)  
            self.loadFileFrame.pack(side='top', expand=True, padx=5, pady=5)
            self.startButton.pack(side='left', expand=True, padx=10, pady=10, ipadx=30, ipady=5)
            self.loadButton.pack(side='left', expand=True, padx=10, pady=10, ipadx=30, ipady=5)
            self.resultsOverallFrame.pack(expand=True, padx=10, pady=0)
            self.resultsHeadingFrame.pack(side='top', expand=True, padx=0, pady=5)
            self.resRankLabel.pack(side='left',expand=True, padx=0, pady=10)
            self.resLabLabel.pack(side='left',expand=True, padx=20, pady=10)
            self.resConfLabel.pack(side='left',expand=True, padx=15, pady=10)
            for n in range(self.numberOfResults):
                self.resultsRowFrame[n].pack(side='top', expand=True, padx=10, pady=5)
                self.resRankEntryLabel[n].pack(side='left', expand=True, padx=5, pady=2.5)
                self.resLabelEntryLabel[n].pack(side='left', expand=True, padx=5, pady=2.5)
                self.resConfEntryLabel[n].pack(side='left', expand=True, padx=10, pady=5)
            self.exitButton.pack(expand=True, padx=10, pady=20, ipadx=30, ipady=5)
            self.exitFrame.pack(side='top', fill="both", expand=True, padx=0, pady=0)
            
        else:
            raise Exception('Unkown overall size {0}!'.format(overall_size))     
            
            
        
    class ourFonts():
        def __init__(self):             
            self.title = font.Font(family='Helvetica', size=24, weight='bold')
            self.titleSmall = font.Font(family='Helvetica', size=12, weight='bold')
            self.button = font.Font(family='Helvetica', size=16)
            self.buttonSmall = font.Font(family='Helvetica', size=8)
            self.labelHead = font.Font(family='Helvetica', size=18)
            self.labelHeadSmall = font.Font(family='Helvetica', size=10)
            self.labelEntry = font.Font(family='Helvetica', size=14,weight='bold')
            self.labelEntrySmall = font.Font(family='Helvetica', size=8,weight='bold')
            self.labelEntryLab = font.Font(family='Courier', size=14)
            self.labelEntryLabSmall = font.Font(family='Courier', size=9)
            
            
    def image_selection(self,overall_size):  
        
          if overall_size == 'normal':
              self.bannerLabel['image'] = self.banner

          elif overall_size == 'small':
              self.bannerLabel['image'] = self.bannerSmall
                 
          else:
              raise Exception('Unkown overall size {0}!'.format(overall_size))       
            

    
    def track_overall_size(self,event):  
        
         change = False
         
         if self.overallSize == 'small':
             if event.width > self.masterFrameSwitchWidth and event.height > self.masterFrameSwitchHeight:
                 change = True 
         elif self.overallSize == 'normal':     
             if event.width < self.masterFrameSwitchWidth or event.height < self.masterFrameSwitchHeight:   
                 change = True 
         else:
              raise Exception('Unkown overall size {0}!'.format(self.overallSize))       
             
         if change:
             self.switch_overall_size() 
            
            
         
    def switch_overall_size(self):  
       
        if self.overallSize == 'small':
            NEW_SIZE = 'normal'
            self.overallSize = 'normal'
        elif self.overallSize == 'normal':
            NEW_SIZE = 'small'
            self.overallSize = 'small'
        else:
            raise Exception('Unkown overall size {0}!'.format(self.overallSize))   
            
        self.image_selection(NEW_SIZE)
        self.apply_styles(NEW_SIZE)
        self.apply_packing(NEW_SIZE)   
        
    def load_audio_file(self):
        self.filename=filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("wav files","*.wav"),("all files","*.*")))
        d = {"sample_rate": 32000, "window_size": 1024,"hop_size": 320, "mel_bins": 64, "fmin": 50, "fmax": 14000, "model_type": "Transfer_Cnn14", "freeze_base": False, "checkpoint_path": "./checkpoints_downloaded/20000_iterations.pth", "audio_path": self.filename, "cuda": False}
        args = SimpleNamespace(**d)    
        self.inference = audio_tagging(args)
        predict_probs, predict_labels = self.inference
        predict_probs, predict_labels = (list(t) for t in zip(*sorted(zip(predict_probs, predict_labels))))
        predict_probs.reverse()
        predict_labels.reverse()
        for n in range(self.numberOfResults):
          self.posthocProb[n].set(predict_probs[n])
          tempStr='{0:<15.15}'.format(predict_labels[n])
          self.resultLabel[n].set(tempStr)
        
            
    def start_stop_recording_and_analysis(self):
        if self.procStatus:
            self.startButton['text'] = 'Start'
            self.demoStyle.configure('StartStop.TButton', background='#7cb669')
            self.demoStyle.map("StartStop.TButton",background=[('active', '#689958')])
            self.procStatus = False
            
            self.aud_detec.stop_detection()
            self.threadAnalysis.join()
        else:
            self.startButton['text'] = 'Stop'
            self.demoStyle.configure('StartStop.TButton', background='#b68169')
            self.demoStyle.map("StartStop.TButton",background=[('active', '#996c58')])
            self.procStatus = True

            self.threadAnalysis = Thread(target=self.run_analysis)
            self.threadAnalysis.start()
            
            
            
    def run_analysis(self):
        
        self.aud_detec.start_detection(self.resultLabel, self.posthocProb)
        
    
    def pause_analysis(self):

        self.aud_detec.stop_detection()
        
            
    def exit_demo(self):
        self.aud_detec.running = False
        self.aud_detec.terminate_detection()
        self.master.destroy()
        

def main(numberOfResults=None): 
     
    if numberOfResults is None and len(sys.argv) > 1:
        numberOfResults = sys.argv[1]
    else: 
        numberOfResults = 3 
    
    root = tk.Tk()
    demo = DemonstratorGUI(master=root,number_of_results=numberOfResults)
    demo.master.title('Making Sense of Sounds - Sound Recognition')
    demo.mainloop()
    
   
if __name__ == '__main__':
    main() 
    
