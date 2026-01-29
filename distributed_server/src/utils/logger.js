/**
 * Logger Utility - Structured logging with colors
 */

const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    dim: '\x1b[2m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m'
};

export class Logger {
    constructor(context = 'App') {
        this.context = context;
    }

    formatTimestamp() {
        return new Date().toISOString();
    }

    formatMessage(level, message, ...args) {
        const timestamp = this.formatTimestamp();
        const prefix = `[${timestamp}] [${this.context}]`;
        
        let formattedArgs = args.map(arg => {
            if (typeof arg === 'object') {
                return JSON.stringify(arg, null, 2);
            }
            return arg;
        }).join(' ');

        return `${prefix} ${level}: ${message} ${formattedArgs}`.trim();
    }

    info(message, ...args) {
        console.log(
            `${colors.cyan}[${this.formatTimestamp()}]${colors.reset} ` +
            `${colors.blue}[${this.context}]${colors.reset} ` +
            `${colors.green}INFO${colors.reset}: ${message}`,
            ...args
        );
    }

    warn(message, ...args) {
        console.warn(
            `${colors.cyan}[${this.formatTimestamp()}]${colors.reset} ` +
            `${colors.blue}[${this.context}]${colors.reset} ` +
            `${colors.yellow}WARN${colors.reset}: ${message}`,
            ...args
        );
    }

    error(message, ...args) {
        console.error(
            `${colors.cyan}[${this.formatTimestamp()}]${colors.reset} ` +
            `${colors.blue}[${this.context}]${colors.reset} ` +
            `${colors.red}ERROR${colors.reset}: ${message}`,
            ...args
        );
    }

    debug(message, ...args) {
        if (process.env.DEBUG === 'true') {
            console.log(
                `${colors.cyan}[${this.formatTimestamp()}]${colors.reset} ` +
                `${colors.blue}[${this.context}]${colors.reset} ` +
                `${colors.dim}DEBUG${colors.reset}: ${message}`,
                ...args
            );
        }
    }
}
