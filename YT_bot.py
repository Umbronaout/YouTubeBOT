"""
Modul containing classes that represent the YouTubeBOT, Channels and videos
Classes contain methods for creating videos and reddit scraping
"""
import os
import json
import random
from time import sleep
from threading import Thread
from moviepy import *
from moviepy.video.tools.subtitles import SubtitlesClip
import Google
from YT_data import *
import praw
import re
import nltk
import numpy as np
from scipy.ndimage import binary_dilation


### Reddit API ###
# Init
reddit_client_id = ''
reddit_client_secret = ''
reddit_user_agent = ''

# Create praw object
reddit = praw.Reddit(client_id= reddit_client_id,
                    client_secret= reddit_client_secret,
                    user_agent= reddit_user_agent)

def clean_text(text):
    """
    Function that removes whitespace, non-ASCII characters
    param: text - text to process
    out: processed text
    This function used to censor the text
    """
    # Remove forbidden characters
    forbidden_characters = [35, 42, 92, 94, 126, 124, 96]

    # Take ASCII characters only
    text = ''.join([_ if ord(_) < 128 or ord(_) not in forbidden_characters else '' for _ in text])
    return text


class YouTubeBOT():
    """
    Main class that manages channels
    """
    def __init__(self) -> None:
        # Code
        self.channels = []
        self.threads = []
        self.importing = False
        self.import_ferq = 24    # Hours how often is reddit scraped
        self.running = True
        self.load_channels()     # Load any saved channels

        # Periodical importing
        self.threads.append(Thread(target= self.import_posts))

        # Start threads
        for thread in self.threads:
            thread.start()

    def import_posts(self):
        """
        Periodic importing - thread target
        """
        def import_from_reddit():
            """
            Imports posts from subreddits and saves them in the 'YT_data.db'
            """

            # Get names of all subredits used by channels
            subreddits = []

            # Get all unique subredits used by channels
            for channel in self.channels:
                subreddits += channel.sub_list
            subreddits = list(set(subreddits))

            # Loop trough all subreddits in list
            for sub in subreddits:
                print(sub)
                # Loop trough the pot 100 "hot" posts
                for submission in reddit.subreddit(sub).hot(limit = 100):
                    # Updating existing entry
                    if post_entry_exists(submission.id):
                        update_entry(table= 'post',
                                        reddit_id= submission.id,
                                        score= int(submission.score),
                                        num_comments = int(submission.num_comments),
                                        ratio= float(submission.upvote_ratio),
                                        NSFW= bool(submission.over_18)
                                        )

                    # If entry doesn't exist create it
                    else:
                        if "update" in str(submission.title).lower():
                            continue
                        else:
                            insert_data(table= 'post',
                                        reddit_id = submission.id,
                                        title= clean_text(submission.title),
                                        sub= sub,
                                        author= str(submission.author),
                                        length= len(submission.selftext),
                                        url= submission.url,
                                        score= int(submission.score),
                                        num_comments = int(submission.num_comments),
                                        ratio= float(submission.upvote_ratio),
                                        NSFW= bool(submission.over_18)
                                        )

                        # Saving the body of the new post as a text file
                        file_name = os.path.join('Posts', str(submission.id) + ".txt")
                        post_body = clean_text(submission.selftext)
                        with open(file_name, 'w') as f:
                            f.write(post_body)
            
            print('Finished import')

        # Running condition for threading
        while self.running:
            # Check if importing
            if self.importing:
                import_from_reddit()
                # Wait for a specifed time, break when shutting down
                for _ in range(60 * 60 * self.import_ferq):
                    if not self.running:
                        break
                    sleep(1)
            # Not importing
            else:
                sleep(1)

    def create_new_channel(self, channel_name, theme, **settings):
        """
        Creates an instance of the Channel class and adds it to its own list
        params: channel_name, theme - minimal req for creating a channel
        """
        # Differentiate between loading (from Settings.json) and creating new with taken name
        if os.path.exists(os.path.join('Channels', channel_name)):
            # Save exists channel is loading
            if os.path.exists(os.path.join('Channels', channel_name, 'Settings.json')):
                # Does not have google acount secrets -> can not upload
                if os.path.exists(os.path.join('Channels', channel_name, 'client_secrets.json')):
                    settings['uploading'] = False

            # Save does not exist -> name taken
            else:
                raise Exception(f'Channel name: "{channel_name}" already taken')

        # Create a channel dir for new channel
        else:
            os.mkdir(os.path.join('Channels', channel_name))

        #Create instance
        new_channel = Channel()

        # Pass req attributes
        new_channel.channel_name = channel_name
        new_channel.theme = theme

        # Pass optional attributes
        for setting, value in settings.items():
            setattr(new_channel, setting, value)

        # Load any finished videos for loaded channels
        new_channel.load_videos()

        # Set channel as running
        new_channel.run()

        # Add channel to the list
        self.channels.append(new_channel)

    def load_channels(self):
        """
        Tries to load all saved channels in 'Channels'
        """
        # Cycle trough everything in 'Channels' and try to load it as if it was a channel dir
        for channel_dir in os.listdir('Channels'):
            # Try skips channel_dirs where Settings.json is missing or corrupted
            try:
                with open(os.path.join('Channels', channel_dir, 'Settings.json'), 'r') as f:
                    channel_data = json.load(f)

                # Create channel entity and load saved data
                self.create_new_channel(**channel_data)
            except:
                pass

    def cleanup(self):
        """
        Properly ends YouTubeBOT
        """
        # Shut down threads
        self.running = False

        # Wait for threads to finish
        for thread in self.threads:
            thread.join()
        self.threads = []

        # Wait for channels to save
        for channel in self.channels:
            channel.save()
        self.channels = []


class Channel():
    """
    Class that represents the bot on a channel level (one bot can have multiple channels)
    Creaes, gives instructions, schedules and deletes instances of the 'video' class
    """
    def __init__(self) -> None:
        # Code
        self.buffer = []            # Video entity queue
        self.buffer_capacity = 10   # Maximum number of videos buffered
        self.threads = []
        self.creating_videos = False
        self.uploading = False
        self.status = 'IDLE'

        # Channel
        self.channel_name = None
        self.theme = None

        # Video
        self.cluster_length = 10            #Cluster = a piece of text displayed at the same time
        self.video_dimensions = (1280, 720)    # 4K: (3840, 2160)
        self.voice = {
            "voice": "en-US-AvaMultilingualNeural"
        }
        self.voice_settings = {
            "Angry": {"voice": "en-US-AvaMultilingualNeural", "pitch": "+5Hz", "rate": "+10%", "volume": "+20%"},
            "Happy": {"voice": "en-US-AvaMultilingualNeural", "pitch": "+5Hz", "rate": "+10%"},
            "Sad": {"voice": "en-US-AvaMultilingualNeural", "pitch": "-2Hz", "rate": "-15%", "volume": "-30%"},
            "Confused": {"voice": "en-US-AvaMultilingualNeural", "pitch": "+0Hz", "rate": "-10%"},
            "Excited": {"voice": "en-US-AvaMultilingualNeural", "pitch": "+10Hz", "rate": "+15%"}
        }

        # YouTube upload
        self.upload_times = []  # list of strings

        # Reddit posts
        self.minimal_post_length = 1E4      #Minimal number of characters of a post to be processed
        self.sub_list = [
            'AITAH',
            'pettyrevenge',
            'tifu',
            'TalesFromRetail',
            'relationship_advice',
            'relationships',
            'AmItheAsshole',
            'TrueOffMyChest',
            'offmychest'
        ]

    def run(self):
        """
        Describes the running beavior of the channel = creating videos, uploading
        """
        # set itself as running
        self.running = True

        def create_videos():
            """
            Periodically checks if a video should be created and adds them if needed - thread target
            """
            # Running condition for threading
            while self.running:
                # Proceeds if creating videos is on
                if self.creating_videos:
                    # Check if there is room in the buffer queue
                    if len(self.buffer) < self.buffer_capacity:
                        # Try to find the next best post - if none wait
                        if find_top_post(self.minimal_post_length, self.sub_list) == None:
                            self.status = 'Can not find a suitable post'
                            sleep(1)
                            continue
                        # If top post found, continue
                        else:
                            reddit_id, title, sub, author = find_top_post(self.minimal_post_length, self.sub_list)

                        # Read the body of the post
                        with open(os.path.join('Posts', reddit_id + '.txt'), 'r') as f:
                            content = f.read()

                        # Create a Video class entity
                        new_video = Video(owner= self, ID= reddit_id)
                        new_video.title= title
                        new_video.sub= sub
                        new_video.author= author
                        new_video.content= content

                        # Create the video and add the entity to buffer
                        self.status = f'Creating video: {reddit_id}'

                        # Start Creating the video
                        new_video.render()

                        # Add a finished video into buffer
                        self.buffer.append(new_video)

                        # Set as idle when video finished
                        self.status = 'IDLE'
                    
                    else:
                        self.status = 'Buffer full'
                        sleep(1)

                # Creating videos turned off
                else:
                    self.status = 'IDLE'
                    sleep(1)

        def upload_videos():
            """
            Periodicaly uploads videos - thread target
            """
            def seconds_until_soonest():
                """
                Nested function to calculate waiting time until the closest upload
                """
                from datetime import datetime, timedelta

                # Current time
                now = datetime.now()

                # Convert the uploading times to datetime objects for today
                uploading_times = [datetime.strptime(time, "%H:%M").replace(year=now.year, month=now.month, day=now.day) for time in self.upload_times]

                # Calculate the time difference in seconds
                time_diffs = [(uploading_time - now).total_seconds() for uploading_time in uploading_times]

                # If all times have passed, add 24 hours to the closest time for tomorrow
                if all(diff <= 0 for diff in time_diffs):
                    closest_time_diff = min([(uploading_time + timedelta(days=1) - now).total_seconds() for uploading_time in uploading_times])
                else:
                    closest_time_diff = min([diff for diff in time_diffs if diff > 0])

                return range(int(closest_time_diff))

            # Running condition for threading
            while self.running:
                # If uploading and there are videos in buffer queue
                if self.uploading and len(self.buffer) > 0:
                    # Wait until the soonest upload time - break if shut down
                    for _ in seconds_until_soonest():
                        if self.running:
                            sleep(1)
                        else:
                            break

                    # Process the first in line video in the buffer queue
                    try:
                        self.buffer[0].upload()
                    except Exception as e:
                        print(f"{e}")
                    # If successful, remove video from queue
                    else:
                        self.buffer.pop(0)
                else:
                    sleep(1)

        # Add threads
        self.threads.append(Thread(target= create_videos))
        self.threads.append(Thread(target= upload_videos))

        # Start threads
        for thread in self.threads:
            thread.start()

    def load_videos(self):
        """
        Tries to load all saved videos (only finished videos are saved)
        Creates Video instance and appends it into the buffer queue
        """
        # Cycle trough everything in own channel dir
        for video in os.listdir(os.path.join('Channels', self.channel_name)):
            # If (video) dir is found
            if os.path.isdir(os.path.join('Channels', self.channel_name, video)):
                try:
                    # Read saved metadata abou the saved viddeo
                    with open(os.path.join('Channels', self.channel_name, video, 'info.json'), 'r') as f:
                        video_data = json.load(f)

                    # Create video object
                    loaded_video = Video(owner= self, ID= video)

                    # Set loaded attrs
                    for setting, value in video_data.items():
                        setattr(loaded_video, setting, value)

                    # Add to channel buffer
                    self.buffer.append(loaded_video)

                # Loading fails for any reason -> file skipped
                except:
                    pass

    def save(self):
        """
        Properly ends videos and own threads
        Saves data into a json to load later
        """
        # Thread condition set
        self.running = False

        # Wait for threads to finish
        for thread in self.threads:
            thread.join()
        self.threads = []

        # Release any owned videos
        for video in self.buffer:
            video.release()
        self.buffer = []

        # Save settings as a json
        channel_data = {}
        for attr, value in self.__dict__.items():
            channel_data[attr] = value
        with open(os.path.join('Channels', self.channel_name, 'Settings.json'), 'w') as f:
            json.dump(channel_data, f, indent=4)


class Video():
    """
    Represents a video created by the bot
    Class with methods used to create and upload the video
    """
    def __init__(self, owner:Channel, ID:str) -> None:
        # Video
        self.title = None
        self.content = None
        self.sub = None
        self.author = None
        self.description = None
        self.upload_time = None
        self.yt_title = None

        # Code
        self.ID = ID
        self.owner = owner
        self.dir = os.path.join('Channels', self.owner.channel_name, self.ID)
        self.finished = False
        self.uploaded = False

        try:
            os.mkdir(self.dir)
        except:
            pass

    def render(self):
        """
        This is the main function that creates the video
        Takes data from the video class and 'YT_data.db'
        Saves full length video in the video dir as name = '(reddit_id).mp4'
        TODO Saves short of the video in the video dir as name = '(reddit_id)_short.mp4'
        """

        def text_to_speech(text_list:list):
            """
            Takes list of texts and creates moviepy AudioClip object
            text list format = [(spoken_text, tag)]
            param: text - text to be processed
            out: audio - MoviePy audio clip with the voice
            saves the audio in the video dir with name = reddit_id.mp3
            assigns clusters to the video as an attr (cluster = pice of text displayed at once)
            The text gets censored during clustering
            The final audio is 
            """
            def is_naughty(word:str, naughty_list:list, lemmatizer, stemmer1, stemmer2):
                """
                Nested function for finding swear words - rule + dictionary based
                param: word - word to be analyzed
                param: naughty_list - list of words that should be specified - loaded from censored_words.txt - word stems is usually enough
                param: lematizer - nltk lemmantizer class object - so it's not inited over and over
                param: stemmer - nltk stemmer class object - so it's not inited over and over
                out: True/False - bool; True = is a swear word, Flase = isn't a swear word
                !!! Loads the words to censor from 'censored_words.txt' !!! in project dir
                """
                word_form_list = []
                # Get rid of any non-alpha chars ()
                word = "".join([char for char in list(word) if char.isalpha()]).lower()

                # Prepare all forms of the word that will be checked
                if word == "":
                    return False
                stem1 = stemmer1.stem(word)
                stem2 = stemmer2.stem(word)

                word_form_list = [word, stem1, stem2]

                for option in ["n", "a", "v", "r", "s"]:
                    lemma = lemmatizer.lemmatize(word, option)
                    word_form_list.append(lemma)

                # Check exceptions
                for word_form in word_form_list:

                    if word_form in [   
                                        "pass", "assur", "assort", "assess", "assert", "assume", "impass", "degass", "assist", "potassium", "assail"
                                        "cumul", "talcum", "curcuma", "cumene",
                                        "fage",
                                        "therapy"
                                    ]:
                        return False

                for word_form in word_form_list:

                    if word_form in naughty_list:
                        return True
                    
                    if len(word_form) >= 6:  # The shortest compound swear word in english is asshat
                        pattern = "|".join(naughty_list)
                        matches = re.findall(pattern, word_form)
                        if matches:
                            for match in matches:
                                if not word_form.startswith(match):
                                    rest = word_form[:word_form.find(match)]
                                else:
                                    rest = word_form.replace(match, "", 1)
                        
                                if len(rest) <= 2 and rest not in ["up", "ty", "ey", "y"]:
                                    continue
                                
                                for word_dict in [naughty_list, words.words()]:
                                    if rest in word_dict:
                                        return True

                return False

            # tts imports
            import asyncio
            import edge_tts

            # text censor imports
            nltk.download('wordnet', quiet=True)  # Required for WordNetLemmatizer
            nltk.download('words', quiet=True)    # Required for the words corpus
            from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
            from nltk.corpus import words

            # prepare for word censoring
            lemmatizer = WordNetLemmatizer()
            stemmer1 = PorterStemmer()
            stemmer2 = LancasterStemmer()

            # Prepare
            output_file = os.path.join(self.dir,  self.ID + ".mp3")
            tupled_text = []
            start = float(0)
            section_boundaries = []

            # Cycle trough text parts
            for idx, text in enumerate(text_list):
                # Start writing or append based on idx
                if idx == 0:
                    mode = "wb"
                else:
                    mode = "ab"
                
                if text[1] in self.owner.voice_settings.keys():
                    voice_setting = self.owner.voice_settings[text[1]]
                else:
                    voice_setting = self.owner.voice
            
                # edge-tts create audio and get wourd boundries for words (time conversion for MoviePy)
                async def amain() -> None:
                    communicate = edge_tts.Communicate(text= text[0], **voice_setting)
                    with open(output_file, mode) as file:
                        async for chunk in communicate.stream():
                            if chunk["type"] == "audio":
                                file.write(chunk["data"])
                            elif chunk["type"] == "WordBoundary":
                                tupled_text.append(((chunk["offset"]/10000000 + start, chunk["offset"]/10000000 + chunk["duration"]/10000000 + start, chunk["duration"]/10000000), chunk["text"], text[1]))     # fromat: ((start, end, duration), word, tag)
    
                # Wait for edge to finish
                asyncio.run(amain())

                end = AudioFileClip(os.path.join(self.dir,  self.ID + ".mp3")).duration # The end of the last tupled word cannot be used becasue there is unbounded silence at start and end of each clip?
                section_boundaries.append(((start, end), text[1]))  # ((start, end), tag)
                start = end

            # Prepare clustering
            self.clusters = []
            cluster = []
            censor_times = []
            with open("censored_words", "r") as f:
                naughty_list = [_.strip() for _ in f.readlines()]

            # Creates cluster that are used to generate subtitles later
            for indx, word in enumerate(tupled_text):
                # Apply censoring
                if is_naughty(word[1], naughty_list, lemmatizer, stemmer1, stemmer2):
                    # Create list of tuples where voice should be muted
                    censor_percentage = 0.5     # How much of the word gets silenced/censored (both voice and text)
                    censor_start = word[0][0] + (1 - censor_percentage) / 2 * word[0][2]
                    censor_end = word[0][1] - (1 - censor_percentage) / 2 * word[0][2]
                    censor_times.append((censor_start, censor_end))

                    # Censor the word in cluster
                    word_len = len(word[1])
                    if word_len == 2:
                        word = (word[0], word[1][0] + "*")
                    else:
                        word = (word[0], word[1][0] + "*" * (word_len - 2) + word[1][-1])
                
                # Cluster
                cluster.append(word)
                if len(' '.join([_[1] for _ in cluster])) >= 10 or\
                        indx == len(tupled_text) - 1 or\
                        tupled_text[indx + 1][1][0].isupper():
                    self.clusters.append(((cluster[0][0][0], cluster[-1][0][1]), ' '.join([_[1] for _ in cluster])))
                    cluster = []

            # The audio has to be saved and then loaded to work properly :(
            audio = AudioFileClip(os.path.join(self.dir,  self.ID + ".mp3"))
            # Moviepy function is bugged this is a work around!
            #audio = audio.subclipped(0,-0.15) #TODO?

            # Apply silences to censor the audio
            for start_time, end_time in censor_times:
                effect = afx.MultiplyVolume(0, start_time=start_time, end_time=end_time)
                audio = audio.with_effects([effect])

            # Returns MoviePy audioclip
            return audio, section_boundaries

        def thumbnail():
            """
            Creates a thumbnail based on a backgraound
            The backround has a square logo in the top left corner
            param: title - title of the video
            param: channel_name - name displayed next to the logo
            out: image_clip - MoviePy image clip of the image with no length
            saves the thumbnail in video dir with name = reddit_id.png
            !!! Loads the background from 'background.png' !!!
            """
            from PIL import Image, ImageDraw, ImageFont
            import textwrap

            logo_dim = 220      # Size of the logo in pixels
            corner = 25         # Size of the texbox corners in pixels
            stroke_width = 15   # Stroke of the character img
            character_width = 520
            stroke_color = random.choice([
                                        # Blues
                                        "dodgerblue", "deepskyblue", "aqua", "skyblue", "mediumblue",
                                        # Oranges/reds
                                        "tomato", "orange", "orangered", "crimson", "darkorange",
                                        # Yellow
                                        "gold", "chocolate"
                                          ])
            font_name = "arial.ttf"
            im = Image.open("background.png")

            ### Add character image with stroke ###
            # Load random character img
            directory = os.path.join("Channels", self.owner.channel_name, "Character_assets")
            entries = os.listdir(directory)
            character_img_path = random.choice([os.path.join(directory, entry) for entry in entries])
            character_img_raw = Image.open(character_img_path)
            character_img = Image.new("RGBA", (character_img_raw.width + 2 * stroke_width, character_img_raw.height + 2 * stroke_width), (0, 0, 0, 0))
            character_img.paste(character_img_raw, (stroke_width, stroke_width), character_img_raw)

            # Add stroke
            # Create a mask of the non-transparent regions
            mask = np.array(character_img.convert("L").point(lambda p: p > 0 and 255))

            # Dilate the mask to create the stroke
            dilated_mask = binary_dilation(mask, iterations=stroke_width)

            # Create an image for the stroke
            stroke_img = Image.new("RGBA", (character_img.width, character_img.height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(stroke_img)

            # Draw the stroke on the dilated mask
            for y in range(dilated_mask.shape[0]):
                for x in range(dilated_mask.shape[1]):
                    if dilated_mask[y, x] and not mask[y, x]:
                        draw.point((x + stroke_width, y + stroke_width), fill=stroke_color)

            # Combine the stroke image with the original image
            character_img_with_stroke = Image.alpha_composite(stroke_img, Image.new("RGBA", stroke_img.size, (0, 0, 0, 0)))
            character_img_with_stroke.paste(character_img, (stroke_width, stroke_width), character_img)

            # Resize
            aspect_ratio = character_img_with_stroke.height / character_img_with_stroke.width
            new_height = int(character_width * aspect_ratio)
            character_img_with_stroke = character_img_with_stroke.resize((character_width, new_height))

            # Position of the character img
            ov_width, ov_height = character_img_with_stroke.size
            position = (1280 - (ov_width * 95) // 100, 720 - (ov_height * 95) // 100)

            # Add character to image
            im.paste(character_img_with_stroke, position, character_img_with_stroke)

            draw = ImageDraw.Draw(im)

            ### Title of the video ###
            # Define the texbox corners
            box = (corner, logo_dim, 1280 - character_width - corner, 720 - corner)
            # Loop variables
            font_size = 200
            size = None

            # While loop to shrink the text until it fits
            while (size is None or size[0] > box[2] - box[0] or size[1] > box[3] - box[1]) and font_size > 0:
                font = ImageFont.truetype(font_name, font_size)
                wrapped_text = textwrap.fill(self.yt_title, width= 30)
                bbox = draw.textbbox((0, 0), wrapped_text, font=font)
                size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                # size = draw.multiline_textsize(wrapped_text, font=font) old version
                font_size -= 1

            # Draw the scaled image into the image
            bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            draw.multiline_text((box[0], - corner + (720 + logo_dim - size[1])/2), wrapped_text, fill="#000", font=font)

            ### Channel name ###
            # Define new textbox for the Channel name
            box = (logo_dim, corner, 1280 - logo_dim, logo_dim)
            # Restart loop variables
            font_size = 200
            size = None

            # While loop to shrink the text until it fits
            while (size is None or size[0] > box[2] - box[0] or size[1] > box[3] - box[1]) and font_size > 0:
                font = ImageFont.truetype(font_name, font_size)
                bbox = draw.textbbox((0, 0), self.owner.channel_name, font=font)    #size = draw.multiline_text(self.owner.channel_name, font=font).size()  old version
                size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                font_size -= 1

            # Draw the scaled text into the image
            bbox = draw.textbbox((0, 0), self.owner.channel_name, font=font)    #size = draw.multiline_text(self.owner.channel_name, font=font).size()  old version
            size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            draw.multiline_text((box[0], (logo_dim - size[1]) / 2 - corner), self.owner.channel_name, fill="#000", font=font)

            im.save(os.path.join(self.dir,  self.ID + ".png"))

            # Create image clip, resized and return it
            image_clip = ImageClip(os.path.join(self.dir,  self.ID + ".png")).resized(width = self.owner.video_dimensions[0] // 4 * 3, height= self.owner.video_dimensions[1] // 4 * 3)

            return image_clip

        def description(music_attributions, video_attributions):
            """
            Creates description based on attributions assigns it as an attr
            TODO generate a summary based on 'self.content'
            """
            # Delimiter used to separate sections
            delimiter = "--------"

            # Description head
            description = [self.yt_title]#, delimiter, 'Original post:', f'www.reddit.com/r/{self.sub}/comments/{self.ID}/'] TODO

            # Add music attributions if any
            if music_attributions != []:
                description += ["\n", "Artist attributions - music", delimiter]
                for m_atrib in music_attributions:
                    description.append(m_atrib)
                description.append(delimiter)

            # Add video attributions if any
            if video_attributions != []:
                description += ["\n", "Artist attributions - video", delimiter]
                for v_atrib in video_attributions:
                    description.append(v_atrib)
                description.append(delimiter)

            # Assign description as an attr
            self.description = '\n'.join(description)
        
        """ Video title """
        write_title_task = """
        I am creating a youtube video about a reddit post.
        I would need to write fitting, attention grabbing title for the video.
        Answer should be just a single title that mentions of the subreddit.
        The subreddit and title of the post:
        """
        title_text = "This is a test Title!" #TODO Google.generate_text(task=write_title_task, text=self.sub + "\n" + self.title, type="text") #
        self.yt_title = title_text

        """ Text """
        # Prepare list with sections of the video
        script = []

        ### Intro ###
        # Intro is generated based on the title and sub of the post - thumbnail is displayed for the duration of the intro - *hopefully* over subs
        write_intro_task = """
        I am creating a youtube video about a reddit post.
        I would need to write the intro for the video.
        My persona is a casual funny woman.
        Answer should be just a few sentences and should mention the title.
        The subreddit and title of the post:
        """
        intro_text = "This is a test intro!" #TODO Google.generate_text(task=write_intro_task, text=self.sub + "\n" + self.title, type="text") #
        script.append((intro_text, "intro"))

        ### Content + Commentary ###
        # Prepare commentary generating task for prompt
        generate_commentary_task = """
        I am creating a youtube video about a reddit post.
        I would need to write a commentary for the post.
        My persona is a casual funny woman.
        Answer should summarize the post with adding own take of the story.
        At the start of every sentence there should be a classification of the sentence in square brackets if applicable choosing from: happy, sad, angry, excited, confused.
        The post I would need the commentary on reads as follows:
        """

        # append part
        script.append((self.content, "content"))
        # generate commentary for the part
        commentary = [("This is a test commentary!", "Angry")] #TODO Google.generate_text(task=generate_commentary_task, text=self.content, type="commentary") #
        # Add each tagged commentary sentence
        for line in commentary:
            script.append(line)

        ### Comments ###
        # Script transition to comments
        comment_transitions = [
            "Now, let's see what the Reddit community thinks about this.",
            "Let's dive into the comments and hear what others have to say.",
            "Time to check out some reactions from fellow Redditors.",
            "Let's explore the comments and see different perspectives.",
            "Now, let's hear from the community and their thoughts.",
            "Let's take a look at what people are saying in the comments.",
            "Now, let's read some of the top comments on this post.",
            "Let's find out what the community has to say about this.",
            "Let's jump into the comments section and see the reactions."
        ]
        script.append((random.choice(comment_transitions), "Excited"))

        # Cycle trough comments to append to the content
        submission = reddit.submission(id=self.ID)

        # Ensure the comments are sorted by 'top'
        submission.comment_sort = 'top'
        submission.comments.replace_more(limit=0)

        # Prepare
        comments = []
        length = len(self.content)
        idx = 1

        def is_unsuitable(comment):
            """
            Nested that contains repeatedly used conditions
            """
            url_pattern = r'(https?://\S+|www\.\S+)'
            return (
                re.search(url_pattern, comment.body) or
                comment.body in ["[removed]", "[deleted]"] or
                comment.collapsed or
                comment.distinguished
            )

        # Cycle through all comments
        for comment in submission.comments:
            if is_unsuitable(comment):
                continue

            # Add comment
            tag = f"Commenter {idx}"
            text = clean_text(comment.body)
            comments.append((text, tag))
            length += len(text)

            # Add OP's reply if there is any
            for reply in comment.replies:
                if not reply.is_submitter or is_unsuitable(reply):
                    continue
                tag = f"OP"
                text = clean_text(reply.body)
                comments.append((text, tag))
                length += len(text)
            
                for reply_reply in reply.replies:
                    if reply_reply.author != comment.author or is_unsuitable(reply_reply):
                        continue
                    tag = f"Commenter {idx}"
                    text = clean_text(reply_reply.body)
                    comments.append((text, tag))
                    length += len(text)

            idx += 1

            if length >= 10000 and len(comments) >= 5:
                break
        
        # Add comments to the script
        script += comments

        ### Outro ###
        # Said at the end of the video
        outros = [
            "Thanks for watching! If you enjoyed this video, don't forget to like, comment, and subscribe for more Reddit stories and reactions.",
            "That's all for today! Let me know your thoughts in the comments below, and I'll see you in the next video.",
            "I hope you enjoyed this story and the comments. Make sure to hit that subscribe button and stay tuned for more!",
            "Thanks for tuning in! Share your opinions in the comments, and don't forget to like and subscribe for more content.",
            "That's it for this video! If you liked it, give it a thumbs up and subscribe for more Reddit tales and discussions.",
            "Thanks for watching! Leave a comment with your thoughts, and be sure to subscribe for more stories and reactions.",
            "I hope you found this story and the comments interesting. Don't forget to like, comment, and subscribe for more videos!",
            "That's a wrap! Let me know what you think in the comments, and don't forget to subscribe for more Reddit content.",
            "Thanks for watching! If you enjoyed this, hit the like button and subscribe for more stories and community reactions.",
            "I hope you enjoyed this video! Leave your thoughts below, and make sure to subscribe for more Reddit stories and comments."
        ]
        script.append((random.choice(outros), "outro"))

        """ Audio """
        ### Voice ###
        # Pass the script to create the voice and section boundaries
        voice, section_boundaries = text_to_speech(script)
        # add silence for outro
        endscreen_duration = 5  # duration of silence in seconds
        fps = voice.fps  # frames per second of the original audio
        silence = AudioArrayClip(np.zeros((endscreen_duration * fps, 2)), fps=fps)
        voice = concatenate_audioclips([voice, silence])

        print(section_boundaries) # TODO
        intro_duration = section_boundaries[0][0][1]

        ### Music ###
        music_clips = []    # List of MoviePy audio clips
        used_music = []     # List of ids of used music tracks
        music_attributions = []  # List of attribution for music
        music_duration = 0  # Tracks current duration of the music_clips duration
        aviable_music = list_media(media= 'stock_music', theme= self.owner.theme) # All music with theme (id, name)

        # Cycle until the music duration >= voice duration
        while music_duration < voice.duration:
            # Get a random viable vmusic track
            media = random.choice([f for f in aviable_music if f[0] not in used_music])

            # Get id and attribution
            used_music.append(media[0])
            if media[2] != '':
                music_attributions.append(media[2])

            # Convert to MoviePy audio clip
            music_clip = AudioFileClip(os.path.join("stock_music", media[1])).with_volume_scaled(0.03)
            #music_clip = music_clip.subclipped(0,-0.15) #Subclip is a workaround to a known bug!

            # Add to current duration and append audio clip (3 to 5 times) to concatenate later
            music_duration += music_clip.duration
            music_clips.append(music_clip)  #afx.AudioLoop(music_clip, random.choice([3,4,5])))

        # Combine music clips and trim the excess
        music = concatenate_audioclips(music_clips)
        music = music.subclipped(0, voice.duration)

        ### Final audio ###
        audio = CompositeAudioClip([voice, music])
        audio = audio.with_effects([afx.AudioFadeOut(endscreen_duration)])

        """ Video """
        ### Intro image ###
        intro_image = thumbnail()
        intro_image = intro_image.with_position('center').with_end(intro_duration - 0.3).with_fps(1)    # 0.3 s = "..."

        ### Section text ###
        # Displayed for comments
        viewed_sections = []
        for section in section_boundaries:
            if "Commenter" in section[1] or "OP" in section[1]:
                viewed_sections.append(section)

        if viewed_sections != []:
            generator = lambda txt, **kwargs: TextClip(text=txt, font='impact.ttf', font_size=75*3//4, stroke_color='black', stroke_width=3*3//4, color="white", margin=(5, 5))
            sections_clip = SubtitlesClip(viewed_sections, make_textclip=generator).with_position(("center", self.owner.video_dimensions[1]//3 - 75*3//4))

        ### Subtitles ###
        generator = lambda txt, **kwargs: TextClip(text=txt, font='impact.ttf', font_size=75, stroke_color='black', stroke_width=3, color="white", margin=(5, 5))
        subtitles = SubtitlesClip(self.clusters, make_textclip=generator).with_position(('center', 'center'))

        ### Background video ###
        video_clips = []    # List of MoviePy video clips
        used_videos = []    # List of used video clip ids
        video_attributions = []  # List of attribution for videos
        video_duration = 0  # Tracks the duration of the video_clips duration
        aviable_video = list_media(media= 'stock_video', theme= self.owner.theme) # All music with theme (id, name)

        # Loop until video duration >= audio duration
        while video_duration < voice.duration:
            # Get a random viable video
            media = random.choice([f for f in aviable_video if f[0] not in used_videos])

            # Append information
            used_videos.append(media[0])
            if media[2] != '':
                video_attributions.append(media[2])

            # Create MoviePy video clip
            video_clip = VideoFileClip(os.path.join("stock_video", media[1])).with_volume_scaled(0)

            # Handle video dims different than final video dims by adding a black background and resizing
            if video_clip.size != self.owner.video_dimensions:
                video_clip = video_clip.resized(height= self.owner.video_dimensions[1]).with_position('center', 'center')
                visuals_background = ColorClip(self.owner.video_dimensions, color=(0, 0, 0)).with_duration(video_clip.duration)
                video_clip = CompositeVideoClip([visuals_background, video_clip])

            #  Add to current duration and append video clip to concatenate later
            video_duration += video_clip.duration
            video_clips.append(video_clip)

        # Combine clips and trim the excess
        background_video = concatenate_videoclips(video_clips)
        background_video = background_video.subclipped(0, voice.duration)

        ### Commentary overlay
        commentary_clips = []
        for section in section_boundaries:
            if section[1] in self.owner.voice_settings.keys():
                # gray overlay
                duration = section[0][1] - section[0][0]
                gray_overlay = ColorClip(self.owner.video_dimensions, color=(0, 0, 0), duration=duration).with_start(section[0][0]).with_opacity(0.6)
                commentary_clips.append(gray_overlay)
                # character
                img_path = os.path.join("Channels", self.owner.channel_name, "Character_assets", section[1] + ".png")
                character_clip = ImageClip(img=img_path, transparent=True, duration=duration).with_start(section[0][0]).resized(width=self.owner.video_dimensions[0]//3, height=self.owner.video_dimensions[1]//3).with_position((0.75, 0.55), relative=True)
                commentary_clips.append(character_clip)

        ### final video (without audio) ###
        videos = [background_video, subtitles, intro_image]
        if viewed_sections != []:
            videos += [sections_clip]
        if commentary_clips != []:
            videos += commentary_clips

        video = CompositeVideoClip(videos)
        video = video.with_effects([vfx.FadeOut(endscreen_duration)])

        ### Short ###
        # centered subtitles
        generator = lambda txt, **kwargs: TextClip(text=txt, font='impact.ttf', font_size=75, stroke_color='black', stroke_width=3, color="white", margin=(5, 5), text_align="center", size=(int(self.owner.video_dimensions[1] * 9 / 16), self.owner.video_dimensions[1]), method='caption')
        subtitles = SubtitlesClip(self.clusters, make_textclip=generator).with_position(('center', 'center'))

        shorts = [background_video, subtitles, intro_image]
        short = CompositeVideoClip(shorts)
        short = short.cropped(x_center=self.owner.video_dimensions[0] // 2, width=int(self.owner.video_dimensions[1] * 9 / 16))
        short.audio = audio
            
        """ Final video and short"""
        ### Final video ###
        video.audio = audio
        #video = video.subclipped(0, 60)   #TODO remove
        #video.write_videofile(self.dir + "\\" + self.ID + ".mp4", fps = 30, logger="bar", codec="libx264", audio_codec="aac")#, ffmpeg_params=[ "-preset", "fast", "-b:v", "10M"], verbose=False, codec="h264_nvenc"

        ### Short ### TODO
        # Clip
        short = short.subclipped(intro_duration, intro_duration + 58)
        # Save
        short.write_videofile(self.dir + "\\short_" + self.ID + ".mp4", fps = 30, logger="bar", codec="libx264", audio_codec="aac")

        ### Write description ###
        description(music_attributions= music_attributions, video_attributions= video_attributions)

        ### Insert video entry into database ###
        insert_data(table='video',
                    used_music= used_music,
                    used_videos= used_videos,
                    reddit_id= str(self.ID),
                    title= str(self.title),
                    length= int(video.duration),
                    channel= str(self.owner.channel_name)
                    )

        # Mark object as finished
        self.finished = True

    def upload(self):
        """
        Method that uploads the finished video
        """
        # Try to upload
        try:
            Google.upload_video(self)
        except Exception as e:
            print(e)
        # If successful mark object as uploaded
        else:
            self.uploaded = True

        # Release the video after upload
        self.release()

    def release(self):
        """
        Deletes video dir if approriate
        """
        def remove_video_dir(location):
            """
            Nested for removing video dirs
            """
            # Remove all files
            for file in os.listdir(location):
                os.remove(os.path.join(location, file))

            # Remove the dir
            os.rmdir(location)

        # If video has been uploaded its dir gets removed
        if self.uploaded:
            remove_video_dir(self.dir)
        else:
            # If the video is only waiting for upload it is saved
            if self.finished:
                video_data = {}
                for attr, value in self.__dict__.items():
                    if attr not in ['clusters', 'owner', "tupled_text"]:
                        video_data[attr] = value

                with open(os.path.join(self.dir, 'info.json'), 'w') as f:
                    json.dump(video_data, f, indent=4)
            else:
                # Delete unfinished video when object deleted prematurely
                remove_video_dir(self.dir)

if __name__ == '__main__':
    # Create an instance of the YouTubeBOT and print all loaded channels and videos
    BOT1 = YouTubeBOT()
    for channel in BOT1.channels:
        print(f"channel:\t{channel.channel_name}")
        for video in channel.buffer:
            print(f"video:\t{video.title}")

    # Create a sample channel
    # BOT1.create_new_channel(channel_name= "Snail Tube", theme= "Snail", minimal_post_length= 0)
    BOT1.create_new_channel(channel_name= "Post Planet", theme= "Minecraft")
    BOT1.cleanup()
