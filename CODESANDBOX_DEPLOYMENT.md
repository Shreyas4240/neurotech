# CodeSandbox Deployment Instructions

To deploy the BCI decoder to CodeSandbox for free (Glitch alternative):

1. Go to https://codesandbox.io
2. Click "Import from GitHub"
3. Import this repository: `Shreyas4240/neurotech`
4. CodeSandbox will automatically:
   - Install dependencies from `requirements.txt`
   - Start the decoder with `decode_live.py --mock`
   - Provide a WebSocket URL

5. Get your Sandbox URL (e.g., `https://your-project.csb.app`)
6. On Vercel, add environment variable:
   - `DECODER_URL`: `your-project.csb.app:8765`

7. Redeploy Vercel

The BCI page will now connect to the real WebSocket on CodeSandbox instead of using demo mode.

Note: Using `--mock` mode for CodeSandbox deployment since we can't run the full LSL stream in the cloud.
