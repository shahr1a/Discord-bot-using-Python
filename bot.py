# bot.py

import os, random, discord, math
import pandas as pd
import numpy as np
import dataframe_image as dfi
import matplotlib.pyplot as plt
from sklearn import linear_model
from word2number import w2n
from datetime import datetime
from discord.ext import commands
from dotenv import load_dotenv

flag: int = 0
counter: int = 0
word_counter = [''][0]
toggle_c: bool = False
toggle_r: bool = False
text_log: str = ""
frame = []
frame_flag = False
m = None
n = []
reg = None
reg_flag = False
cleaning = False

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    for guild in bot.guilds:
        if guild.name == GUILD:
            break

    print(
        f'{bot.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )


@bot.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(
        f'Hi {member.name}, Welcome To Python Project Server!'
    )


@bot.command(name='chat', help='- Toggles Chat Bot')
async def toggle_chat(ctx):
    global toggle_c
    if not toggle_c:
        toggle_c = True
        response = 'Chat Mode Turned On'
    else:
        toggle_c = False
        response = 'Chat Mode Turned Off'
    await ctx.send(response)


@bot.command(name='reader', help='- Toggles Bot To Read & Save Texts')
async def toggle_read(ctx):
    global toggle_r
    if not toggle_r:
        toggle_r = True
        response = 'Read Mode Turned On'
    else:
        toggle_r = False
        response = 'Read Mode Turned Off'
    await ctx.send(response)


@bot.command(name='list', help='- Provides List of Members')
async def member_list(ctx):
    for guild in bot.guilds:
        if guild.name == GUILD:
            break

    members = '\n - '.join([member.name for member in guild.members])
    response = f'Server Members:\n - {members}'
    await ctx.send(response)


@bot.command(name='wordGraph', help="- Shows Server's Used Words Graph")
async def plot_graph(ctx):
    df = pd.read_csv('words.csv')
    x = df.sort_values(by='Count', ascending=False)['Word'].head(10).values
    y = df.sort_values(by='Count', ascending=False)['Count'].head(10).values
    image = discord.File("words.png")
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(10)
    plt.title("Words Demographic")
    plt.xlabel('Words')
    plt.ylabel('Word Count')
    plt.bar(x, y)
    plt.xticks(rotation=90)
    plt.savefig("words.png")
    plt.close()
    await ctx.send(file=image)


@bot.command(name='load', help="- Loads Dataframe")
async def df_loader(ctx):
    global frame, cleaning, frame_flag
    frame = pd.read_csv('hiring.csv')
    response = "Dataframe Loaded"
    cleaning = False
    frame_flag = True
    await ctx.send(response)


@bot.command(name='unload', help="- Unloads Dataframe")
async def df_unloader(ctx):
    global frame, cleaning, reg, m, n, frame_flag, reg_flag
    if frame_flag:
        frame = []
        reg = None
        m = None
        n = []
        response = "Dataframe Unloaded"
        cleaning = False
        reg_flag = False
        frame_flag = False
        await ctx.send(response)
    else:
        response = "No Dataframe Loaded"
        await ctx.send(response)


@bot.command(name='clean', help="- Cleans Dataframe")
async def df_cleaner(ctx):
    global frame, cleaning, frame_flag
    if frame_flag:
        frame.rename(columns={"salary($)": 'salary'}, inplace=True)
        frame.rename(columns={"test_score(out of 10)": 'test_score'}, inplace=True)
        frame.rename(columns={"interview_score(out of 10)": 'interview_score'}, inplace=True)
        frame.experience = frame.experience.fillna("zero")
        frame.experience = frame.experience.apply(w2n.word_to_num)
        mean_experience = math.floor(frame['experience'].mean())
        frame.loc[frame.experience == 0, 'experience'] = mean_experience
        mean_t_score = math.floor(frame['test_score'].mean())
        frame['test_score'] = frame['test_score'].fillna(mean_t_score)
        mean_i_score = math.floor(frame['interview_score'].mean())
        frame['interview_score'] = frame['interview_score'].fillna(mean_i_score)
        response = "Dataframe Cleaned"
        cleaning = True
        await ctx.send(response)
    else:
        response = "No Dataframe Loaded"
        await ctx.send(response)


@bot.command(name='applyRegression', help='- Applies Regression On Dataframe')
async def df_regression(ctx):
    global frame, m, n, reg, cleaning, frame_flag, reg_flag
    if frame_flag:
        if cleaning:
            reg = linear_model.LinearRegression()
            reg.fit(frame[['experience', 'test_score', 'interview_score']], frame['salary'])
            n = []
            for i in range(50):
                m = reg.predict([[random.randint(1, 15), random.randint(5, 10), random.randint(5, 10)]])
                n = np.append(n, m)
            response = "Regression Applied"
            reg_flag = True
            await ctx.send(response)
        else:
            response = "Regression Cannot Be Applied On Unclean Dataframe"
            await ctx.send(response)
    else:
        response = "No Dataframe Loaded"
        await ctx.send(response)


@bot.command(name='showDF', help='- Shows Dataframe')
async def df_show_dataframe(ctx):
    global frame, frame_flag
    if frame_flag:
        # frame_image = frame.style.background_gradient()
        frame_short = frame.head(11)
        image = discord.File("dataframe.png")
        data = discord.File("dataframe.csv")
        dfi.export(frame_short, "dataframe.png")
        frame.to_csv('dataframe.csv', mode='w', index=False)
        await ctx.send(file=image)
        await ctx.send(file=data)

    else:
        response = "No Dataframe Loaded"
        await ctx.send(response)


@bot.command(name='showGraph', help='- Shows Dataframe as Graph')
async def df_show_graph(ctx):
    global frame, n, cleaning, frame_flag, reg_flag
    if frame_flag:

        if cleaning:
            # %matplotlib inline
            plt.xlabel('Salary')
            plt.ylabel('Experience')
            image = discord.File("hiring.png")
            plt.scatter(frame.salary, frame.experience, color='red', marker='.')
            if reg_flag:
                plt.scatter(n, frame.experience, color='blue', marker='+')
            plt.savefig("hiring.png")
            plt.close()
            await ctx.send(file=image)

        else:
            response = "Dataframe Needs To Be Cleaned First!"
            await ctx.send(response)

    else:
        response = "No Dataframe Loaded To Show Graph"
        await ctx.send(response)


@bot.event
async def on_message(message):
    global toggle_c, toggle_r, text_log, counter, flag

    # Chat Logger
    time = datetime.now()
    c_time = time.strftime("[%H:%M]")
    text_log = message.content
    logger = open("chatLog.txt", "a")
    logger.write(f'{message.author.name}' + c_time + ' - ' + text_log)
    logger.write("\n")
    logger.close()

    # User Checker
    if message.author == bot.user:
        return

    # Word Listing
    if toggle_r:
        for item in message.content.lower().split():
            if not os.path.isfile('words.csv'):
                listing = pd.DataFrame([[item, 1]], columns=['Word', 'Count'])
                listing.to_csv('words.csv', mode='a', index=False)
            else:
                df = pd.read_csv('words.csv', usecols=['Word', 'Count'])
                for word in df["Word"]:
                    if item == word:
                        count = df.iloc[counter, 1]
                        df.at[counter, 'Count'] = count + 1
                        df.to_csv('words.csv', mode='w', index=False)
                        flag = 1
                    counter = counter + 1

                counter = 0
                if flag == 0:
                    listing = pd.DataFrame([[item, 1]])
                    listing.to_csv('words.csv', mode='a', index=False, header=False)
                flag = 0

    # Chat Service
    if toggle_c == True:
        if message.author.nick == None:
            welcome_quotes = [
                f'Hello {message.author.name}',
                f'Hi {message.author.name}',
                f"What's Up {message.author.name}"
            ]
        else:
            welcome_quotes = [
                f'Hello {message.author.nick}',
                f'Hi {message.author.nick}',
                f"What's Up {message.author.nick}"
            ]

        if message.content.lower() in ['hello', 'hi']:
            response = random.choice(welcome_quotes)
            await message.channel.send(response)

    await bot.process_commands(message)


bot.run(TOKEN)
