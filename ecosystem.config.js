module.exports = {
  apps: [
    {
      name: 'AscendingIntelligenceTouchDesigner',
      script: 'C:/Program Files/Derivative/TouchDesigner/bin/TouchDesigner.exe',
      args: [
        'C:/Users/Dev/Documents/Projects/KarenPalmer/AscendingIntelligence/touchdesigner/main.toe',
        '-perform',
        //'-exit'
      ],
      cwd: 'C:/Users/Dev/Documents/Projects/KarenPalmer/AscendingIntelligence/touchdesigner/',
      exec_mode: 'fork',
      interpreter: 'none',
      autorestart: true,
      autorestart_delay: 15000,
      watch: false,
      max_memory_restart: '4G',
    },
  ],
};