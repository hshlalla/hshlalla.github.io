# AI Agent Connection Guide

The frontend UI for the **DeepSound Agent** is now ready. To make it functional, you have two main options:

## Option 1: Dify / Chatbase (Recommended for Security)
These services provide a secure, hosted backend for your AI agent.
1. Create an account on [Dify.ai](https://dify.ai) or [Chatbase.co](https://chatbase.co).
2. Upload your technical documents or provide your site URL for the agent to learn.
3. Replace the `ai-chat-widget.html` contents with the **Embed Code** provided by the service.

## Option 2: Custom OpenAI Integration (Developer Choice)
If you want full control and have an OpenAI API key:
1. Update `_includes/ai-chat-widget.html`.
2. Replace the `sendMessage()` placeholder with a `fetch` call to your own serverless function (e.g., Vercel Functions) that holds your API key.
3. **DO NOT** put your API key directly in the HTML file, as it will be public on GitHub.

### Next Steps for "Work Dashboard"
To make the dashboard "Alive":
- Update your `_posts` regularly.
- You can add a `_data/stats.yml` file to feed real numbers into the "Performance" module.
