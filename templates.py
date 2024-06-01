from helpers import load_image_as_base64

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #1b1b1b
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
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
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

'''
    
captain_img = load_image_as_base64('./assets/captain.png')
user_img = load_image_as_base64('./assets/user.png')

bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/png;base64,{captain_img}" />
    </div>
    <div id="content">
        <div class="message">{{{{MSG}}}}</div>
        <div id="images">{{{{IMAGES}}}}</div>
    </div>
</div>
'''

user_template = f'''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/png;base64,{user_img}" />
    </div>    
    <div id="content">
        <div class="message">{{{{MSG}}}}</div>
        <div id="images">{{{{IMAGE}}}}</div>
    </div>
</div>
'''