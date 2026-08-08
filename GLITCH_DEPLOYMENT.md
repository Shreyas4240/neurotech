# Glitch Deployment Instructions

To deploy the BCI decoder to Glitch for free:

1. Go to https://glitch.com
2. Click "New Project" → "Import from GitHub"
3. Import this repository: `Shreyas4240/neurotech`
4. Glitch will automatically:
   - Install dependencies from `requirements.txt`
   - Start the decoder with `decode_live.py --mock`
   - Provide a WebSocket URL

5. Get your Glitch project URL (e.g., `https://your-project.glitch.me`)
6. On Vercel, add environment variable:
   - `DECODER_URL`: `your-project.glitch.me:8765`

7. Redeploy Vercel

The BCI page will now connect to the real WebSocket on Glitch instead of using demo mode.

Note: Using `--mock` mode for Glitch deployment since we can't run the full LSL stream in the cloud.
