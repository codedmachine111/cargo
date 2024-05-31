from helpers import load_image_as_base64

# Custom CSS being used for UI
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
}

.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    border: 1px solid #ccc;
    background-color: #0f0f0f
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}

.chat-history {
    max-height: 60vh;
    overflow-y: scroll;
    padding: 1rem;
    border: 1px solid #ccc;
}

#content{
    display: flex;
    flex-direction: column;
}

#images{
    display: flex;
    border-radius: 0.5rem;
    flex-direction: row;
    gap: 3;
}

#res_img{
    width: 50%;
    margin: 0.5rem;
}

#message{
    width: 100%;
    padding: 1rem;
    text-align: left;
}
'''
    
captain_img = load_image_as_base64('./assets/captain.png')
user_img = load_image_as_base64('./assets/user.png')

# Chat message HTML for Bot
bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/png;base64,{captain_img}" />
    </div>
    <div id="content">
        <div id="message">{{{{MSG}}}}</div>
        <div id="images">{{{{IMAGES}}}}</div>
    </div>
</div>
'''

# Chat message HTML for User
user_template = f'''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/png;base64,{user_img}" />
    </div>    
    <div id="content">
        <div id="message">{{{{MSG}}}}</div>
        <div id="images">{{{{IMAGE}}}}</div>
    </div>
</div>
'''