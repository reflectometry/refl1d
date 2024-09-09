export class Signal {
    name: string;
    buffer: SharedArrayBuffer;

    constructor(name: string, buffer: SharedArrayBuffer = new SharedArrayBuffer(1)) {
        this.name = name;
        this.buffer = buffer;      
    }

    is_set() {
        return new Int8Array(this.buffer)[0] === 1;
    }

    set() {
        console.log("setting signal:", this.name);
        new Int8Array(this.buffer)[0] = 1;
    }

    clear() {
        console.log("clearing signal:", this.name);
        new Int8Array(this.buffer)[0] = 0;
    }

    wait(timeout?: number) {
        const signal = this;
        return new Promise((resolve, reject) => {
            const interval = setInterval(() => {
                if (signal.is_set()) {
                    clearInterval(interval);
                    resolve(true);
                }
            }, 100);
            if (timeout === undefined) {
                return;
            }
            setTimeout(() => {
                clearInterval(interval);
                reject("Timeout");
            }, timeout);
        });
    }
}
