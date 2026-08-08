# Replit Deployment Instructions

To deploy the BCI decoder to Replit for free:

1. Go to https://replit.com
2. Click "Create Repl" → "Import from GitHub"
3. Import this repository: `Shreyas4240/neurotech`
4. Replit will automatically:
   - Install dependencies from `requirements.txt`
   - Start the decoder with `decode_live.py --mock`
   - Provide a WebSocket URL

5. Get your Repl URL (e.g., `https://your-project.replit.app`)
6. On Vercel, add environment variable:
   - `DECODER_URL`: `your-project.replit.app:8765`

7. Redeploy Vercel

The BCI page will now connect to the real WebSocket on Replit instead of using demo mode.

Note: Using `--mock` mode for Replit deployment since we can't run the full LSL stream in the cloud.
